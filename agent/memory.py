import os
import json
import pickle
import string
import re
import math
import asyncio
import uuid
import threading
import time
from collections import defaultdict,Counter
from datetime import datetime,timedelta
from bot_config import config
from logger import logging

MAX_TOKENS=config.search.max_tokens
CONTEXT_CHUNKS=config.search.context_chunks

# ── learned-stopword knobs ───────────────────────────────────────────────────
_RELEARN_EVERY          = 50    # relearn after this many inserts
_GLOBAL_STOP_THRESHOLD  = 0.70  # word in ≥70 % of ALL memories → global stop
_USER_STOP_THRESHOLD    = 0.80  # word in ≥80 % of a user's memories → user stop
_MIN_GLOBAL_MEMORIES    = 20    # corpus must be at least this large to learn
_MIN_USER_MEMORIES      = 15    # per-user corpus floor

class AtomicSaver:
    def __init__(self,path,get_state,debounce=0.3,logger=None,on_save=None):
        self.path=path; self.get_state=get_state; self.ev=threading.Event(); self.debounce=debounce; self.logger=logger; self.on_save=on_save
        threading.Thread(target=self._run,name="mem.save",daemon=True).start()
    def request(self):
        if self.logger: self.logger.info("mem.save.req")
        self.ev.set()
    def save_now(self):
        try:
            d=self.get_state(); tmp=self.path+'.tmp'
            os.makedirs(os.path.dirname(self.path),exist_ok=True)
            with open(tmp,'wb') as f:
                pickle.dump(d,f,protocol=5); f.flush(); os.fsync(f.fileno()); sz=f.tell()
            os.replace(tmp,self.path)
            try: dfd=os.open(os.path.dirname(self.path),os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
            except: pass
            if self.logger: self.logger.info(f"mem.save.ok path={self.path} bytes={sz}")
            if self.on_save:
                try: self.on_save(self.path)
                except Exception: pass
        except Exception as e:
            if self.logger: self.logger.error(f"mem.save.err path={self.path} msg={e!r}")
    def _run(self):
        while True:
            self.ev.wait(); time.sleep(self.debounce); self.ev.clear()
            if self.logger: self.logger.info("mem.save.run")
            self.save_now()

class TempJanitor:
    def __init__(self,cache_manager,interval=900,force=False,logger=None):
        self.cm=cache_manager; self.itv=interval; self.force=force; self.logger=logger
        threading.Thread(target=self._run,name=f"temp.janitor[{self.cm.bot_name}]",daemon=True).start()
    def _run(self):
        while True:
            time.sleep(self.itv)
            try:
                if self.logger: self.logger.info(f"temp.janitor.run bot={self.cm.bot_name}")
                self.cm.cleanup_temp_files(force=self.force)
            except Exception as e:
                if self.logger: self.logger.error(f"temp.janitor.err msg={e!r}")


class CacheManager:
    _janitors={}
    _jlock=threading.Lock()

    def __init__(self,bot_name,temp_file_ttl=3600):
        self.bot_name=bot_name
        self.temp_file_ttl=temp_file_ttl
        self.base_cache_dir=os.path.join('cache',self.bot_name)
        self.logger=logging.getLogger(f'bot.{self.bot_name}.cache')
        os.makedirs(self.base_cache_dir,exist_ok=True)
        with CacheManager._jlock:
            if bot_name not in CacheManager._janitors:
                CacheManager._janitors[bot_name]=TempJanitor(self,interval=900,logger=self.logger)

    def get_cache_dir(self,cache_type):
        d=os.path.join(self.base_cache_dir,cache_type); os.makedirs(d,exist_ok=True); return d
    def get_temp_dir(self):
        return self.get_cache_dir('temp')
    def get_user_temp_dir(self,user_id):
        d=os.path.join(self.get_temp_dir(),str(user_id)); os.makedirs(d,exist_ok=True); return d
    def create_temp_file(self,user_id,prefix=None,suffix=None,content=None):
        file_id=str(uuid.uuid4()); ts=datetime.now().strftime('%Y%m%d_%H%M%S'); parts=[]
        if prefix: parts.append(prefix)
        parts.extend([ts,file_id]); fn='_'.join(parts)
        if suffix: fn+=suffix
        p=os.path.join(self.get_user_temp_dir(user_id),fn)
        mode='wb' if isinstance(content,bytes) else 'w'; enc=None if isinstance(content,bytes) else 'utf-8'
        with open(p,mode,encoding=enc) as f:
            if content is not None: f.write(content)
        self.logger.info(f"temp.create user={user_id} path={p}")
        meta={'created_at':datetime.now().isoformat(),'user_id':user_id,'file_id':file_id,'original_filename':fn}
        with open(f"{p}.meta",'w') as f: json.dump(meta,f)
        return p,file_id
    def get_temp_file(self,user_id,file_id):
        utd=self.get_user_temp_dir(user_id)
        for fn in os.listdir(utd):
            if fn.endswith('.meta'):
                mp=os.path.join(utd,fn)
                with open(mp,'r') as f: md=json.load(f)
                if md['file_id']==file_id and md['user_id']==user_id:
                    fp=mp[:-5]
                    if os.path.exists(fp): return fp
        return None
    def cleanup_temp_files(self,force=False):
        now=datetime.now(); td=self.get_cache_dir('temp')
        for uid in os.listdir(td):
            utd=os.path.join(td,uid)
            if not os.path.isdir(utd): continue
            for fn in os.listdir(utd):
                if fn.endswith('.meta'): continue
                fp=os.path.join(utd,fn); mp=f"{fp}.meta"
                try:
                    if os.path.exists(mp):
                        with open(mp,'r') as f: md=json.load(f)
                        ca=datetime.fromisoformat(md['created_at']); age=now-ca
                        if force or age>timedelta(seconds=self.temp_file_ttl): self.remove_temp_file(md['user_id'],md['file_id'])
                    else:
                        age=now-datetime.fromtimestamp(os.path.getctime(fp))
                        if force or age>timedelta(seconds=self.temp_file_ttl): os.remove(fp)
                except Exception as e:
                    self.logger.error(f"temp.cleanup.error path={fp} msg={e}")
            if not os.listdir(utd): os.rmdir(utd)
    def remove_temp_file(self,user_id,file_id):
        fp=self.get_temp_file(user_id,file_id)
        if not fp: return
        if os.path.exists(fp): os.remove(fp)
        mp=f"{fp}.meta"
        if os.path.exists(mp): os.remove(mp)
        self.logger.info(f"temp.remove user={user_id} file_id={file_id}")

class UserMemoryIndex:
    def __init__(self,cache_type,max_tokens=MAX_TOKENS,context_chunks=CONTEXT_CHUNKS,logger=None):
        parts=cache_type.split('/')
        if len(parts)>=2: self.bot_name=parts[0]; cache_subtype=parts[-1]
        else: self.bot_name='default'; cache_subtype=cache_type
        self.cache_manager=CacheManager(self.bot_name)
        self.cache_dir=self.cache_manager.get_cache_dir(cache_subtype)
        self.max_tokens=max_tokens; self.context_chunks=context_chunks
        self.inverted_index=defaultdict(list); self.memories=[]; self.user_memories=defaultdict(list)
        self.stopwords=set(['the','a','an','and','or','but','nor','yet','so','in','on','at','to','for','of','with','by','from','up','about','i','you','he','she','it','we','they','me','him','her','us','them','is','are','was','were','be','been','have','has','had','can','could','may','might','must','shall','should','will','would','this','that','these','those'])
        self._global_stops: set = set()
        self._user_stops:  dict = {}   # user_id → set[str]
        self._inserts_since_relearn: int = 0
        self.logger=logger or logging.getLogger('bot.default')
        self._mut=threading.RLock()
        self._cache_mtime=0.0
        self._saver=AtomicSaver(os.path.join(self.cache_dir,'memory_cache.pkl'),self._snapshot,debounce=0.3,logger=self.logger,on_save=self._on_save_complete)
        self.load_cache()
    def _snapshot(self):
        with self._mut:
            return {
                'inverted_index': {k: list(v) for k, v in self.inverted_index.items()},
                'memories': list(self.memories),
                'user_memories': {k: list(v) for k, v in self.user_memories.items()}
            }
    def clean_text(self,text):
        text=text.replace("<|endoftext|>","").replace("<|im_start|>","").replace("<|im_end|>","").lower()
        text=text.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
        out=[]
        for w in text.split():
            if not w or w in self.stopwords or w in self._global_stops:continue
            # Drop short pure-digit noise; keep 4-digit tokens (years) and alphanumeric mixed
            if w.isdigit() and len(w)!=4:continue
            out.append(w)
        return ' '.join(out)
    def _on_save_complete(self,path):
        try: self._cache_mtime=os.path.getmtime(path)
        except OSError: pass

    def _relearn_stops(self, user_id=None):
        """Derive stopwords from the current index DF distributions.

        Global stops: words present in ≥_GLOBAL_STOP_THRESHOLD of all memories.
        User stops:   words present in ≥_USER_STOP_THRESHOLD of a specific
                      user's memories (or all users when user_id is None).
        Both are transient — never persisted; recomputed from the index on load.
        Must be called with self._mut held.
        """
        total = sum(1 for m in self.memories if m is not None)
        if total >= _MIN_GLOBAL_MEMORIES:
            cutoff = total * _GLOBAL_STOP_THRESHOLD
            self._global_stops = {
                w for w, ids in self.inverted_index.items()
                if len(ids) >= cutoff
            }

        targets = [user_id] if user_id else list(self.user_memories.keys())
        for uid in targets:
            umids  = self.user_memories.get(uid, [])
            u_total = len(umids)
            if u_total < _MIN_USER_MEMORIES:
                self._user_stops.pop(uid, None)
                continue
            uid_set  = set(umids)
            cutoff_u = u_total * _USER_STOP_THRESHOLD
            self._user_stops[uid] = {
                w for w, ids in self.inverted_index.items()
                if len(set(ids) & uid_set) >= cutoff_u
            }

    def _check_and_reload(self):
        """Reload from disk if another process has updated the cache. Must be called with self._mut held."""
        cf=os.path.join(self.cache_dir,'memory_cache.pkl')
        try: mtime=os.path.getmtime(cf)
        except OSError: return
        if mtime<=self._cache_mtime: return
        try:
            with open(cf,'rb') as f: d=pickle.load(f)
            self.inverted_index=defaultdict(list,d.get('inverted_index',{}))
            self.memories=d.get('memories',[])
            self.user_memories=defaultdict(list,d.get('user_memories',{}))
            self._cache_mtime=mtime
            self.logger.info(f"mem.reload.ok mtime={mtime}")
        except Exception as e:
            self.logger.warning(f"mem.reload.err: {e}")

    def add_memory(self,user_id,memory_text):
        with self._mut:
            self._check_and_reload()
            mid=len(self.memories); self.memories.append(memory_text); self.user_memories[user_id].append(mid)
            for w in self.clean_text(memory_text).split(): self.inverted_index[w].append(mid)
            self._inserts_since_relearn+=1
            if self._inserts_since_relearn>=_RELEARN_EVERY:
                self._relearn_stops(user_id=user_id)
                self._inserts_since_relearn=0
        self.logger.info(f"mem.add user={user_id} mid={mid}")
        self._saver.request()
    async def add_memory_async(self,user_id,memory_text):
        loop=asyncio.get_event_loop(); await loop.run_in_executor(None,lambda: self.add_memory(user_id,memory_text))
    def _compact(self):
        remap={}; new_mems=[]
        for old_id,m in enumerate(self.memories):
            if m is not None: remap[old_id]=len(new_mems); new_mems.append(m)
        self.memories=new_mems
        for uid in list(self.user_memories.keys()):
            self.user_memories[uid]=[remap[mid] for mid in self.user_memories[uid] if mid in remap]
            if not self.user_memories[uid]: del self.user_memories[uid]
        new_idx=defaultdict(list)
        for w,pl in self.inverted_index.items():
            remapped=[remap[mid] for mid in pl if mid in remap]
            if remapped: new_idx[w]=remapped
        self.inverted_index=new_idx
        self._relearn_stops()
    def clear_user_memories(self,user_id):
        with self._mut:
            if user_id not in self.user_memories: return
            kill=set(self.user_memories[user_id]); ct=len(kill)
            for mid in kill: self.memories[mid]=None
            del self.user_memories[user_id]
            self._compact()
        self.logger.info(f"mem.clear user={user_id} removed={ct}")
        self._saver.request()
    def search(self,query,k=5,user_id=None,dedup_threshold=0.95):
        cq=self.clean_text(query)
        user_stops=self._user_stops.get(user_id,set()) if user_id else set()
        qws=[w for w in cq.split() if w not in user_stops]
        with self._mut:
            self._check_and_reload()
            scores=Counter(); total=len([m for m in self.memories if m is not None])
            rel=self.user_memories.get(user_id,[]) if user_id else range(len(self.memories))
            k1=1.2; b=0.75; dlen={}; tl=0; vc=0
            for mid in rel:
                if mid<len(self.memories) and self.memories[mid] is not None:
                    dl=len(self.memories[mid].split()); dlen[mid]=dl; tl+=dl; vc+=1
            avg=tl/vc if vc>0 else 1.0; rset=set(rel) if user_id else None
            for w in qws:
                pl=self.inverted_index.get(w,[])
                if not pl: continue
                post=[m for m in pl if (m in rset)] if rset else pl
                if not post: continue
                tfc=Counter(post); df=len(tfc); idf=math.log((total-df+0.5)/(df+0.5)+1.0)
                for mid,tf in tfc.items():
                    dl=dlen.get(mid,1); ln=k1*((1-b)+b*(dl/avg)); wt=idf*((k1+1)*tf)/(ln+tf); scores[mid]+=wt
            for mid,score in list(scores.items()):
                dl=dlen.get(mid,1); scores[mid]=score/max(1e-9,math.log(1+dl))
            mx=max(scores.values()) if scores else 1.0
            for mid in list(scores.keys()): scores[mid]/=mx
            sm=sorted(scores.items(),key=lambda x:x[1],reverse=True)
            # Retrieval is bounded by candidate count, not by the original text
            # size.  Hippocampus compresses candidates for embedding and the
            # final prompt builder applies the real context-token budget.
            res=[]; seen=set()
            for mid,sc in sm:
                m=self.memories[mid]
                if m is None: continue
                cm=self.clean_text(m); dup=False
                for s in seen:
                    if self._calculate_similarity(cm,s)>dedup_threshold: dup=True; break
                if not dup:
                    res.append((m,sc)); seen.add(cm)
                    if len(res)>=k: break
        self.logger.info(f"mem.search q='{query[:256]}' got={len(res)}")
        return res
    def _calculate_similarity(self,t1,t2):
        def grams(t,n=3): return set(t[i:i+n] for i in range(len(t)-n+1))
        g1=grams(t1); g2=grams(t2); inter=len(g1&g2); uni=len(g1|g2); return inter/uni if uni>0 else 0
    def save_cache(self):
        self.logger.info("mem.save.force")
        self._saver.save_now()
    def _save_cache_sync(self):
        cf=os.path.join(self.cache_dir,'memory_cache.pkl'); tmp=cf+'.tmp'
        d={'inverted_index':self.inverted_index,'memories':self.memories,'user_memories':self.user_memories}
        os.makedirs(self.cache_dir,exist_ok=True)
        with open(tmp,'wb') as f:
            pickle.dump(d,f,protocol=5); f.flush(); os.fsync(f.fileno()); sz=f.tell()
        os.replace(tmp,cf)
        try: dfd=os.open(self.cache_dir,os.O_DIRECTORY); os.fsync(dfd); os.close(dfd)
        except: pass
        self.logger.info(f"mem.save.sync.ok path={cf} bytes={sz}")
    def load_cache(self,cleanup_orphans=False,cleanup_nulls=True):
        cf=os.path.join(self.cache_dir,'memory_cache.pkl')
        if os.path.exists(cf):
            with open(cf,'rb') as f: d=pickle.load(f)
            try: self._cache_mtime=os.path.getmtime(cf)
            except OSError: pass
            with self._mut:
                self.inverted_index=defaultdict(list,d.get('inverted_index',{}))
                self.memories=d.get('memories',[])
                self.user_memories=defaultdict(list,d.get('user_memories',{}))
                if cleanup_orphans:
                    allm=set(); [allm.update(m) for m in self.user_memories.values()]
                    orph=[i for i,m in enumerate(self.memories) if i not in allm and m is not None]
                    if orph: self.logger.warning(f"orphans={len(orph)}")
                has_nulls=any(m is None for m in self.memories)
                if has_nulls and cleanup_nulls:
                    self._compact()   # _compact calls _relearn_stops internally
                    self.logger.info("mem.compact nulls.removed"); self.save_cache()
                else:
                    self._relearn_stops()
                tot=len(self.memories); act=len([m for m in self.memories if m is not None])
                self.logger.info(f"mem.load totals={tot} active={act} users={len(self.user_memories)} vocab={len(self.inverted_index)} global_stops={len(self._global_stops)}")
            return True
        return False
    async def search_async(self,query,k=32,user_id=None,dedup_threshold=0.95):
        loop=asyncio.get_event_loop(); return await loop.run_in_executor(None,lambda: self.search(query,k,user_id,dedup_threshold))

class RepoIndex:
    def __init__(self,bot_name,branch='main'):
        self.cm=CacheManager(bot_name)
        self.cache_dir=self.cm.get_cache_dir('repo_index')
        os.makedirs(self.cache_dir,exist_ok=True)
        self.branch=branch; self.repo_index=defaultdict(set); self.load_cache()
    def index_file(self,file_path,content,branch=None):
        b=branch or self.branch; self.repo_index[b].add(file_path)
        fp=os.path.join(self.cache_dir,f'{b}_{file_path.replace("/","_")}.txt')
        with open(fp,'w',encoding='utf-8') as f: f.write(content)
    def search_repo(self,query,branch=None,limit=5):
        b=branch or self.branch; q=set(re.findall(r'\b\w{3,}\b',query.lower()))
        if not q: return []
        res=[]
        for fp in self.repo_index.get(b,set()):
            try:
                with open(os.path.join(self.cache_dir,f'{b}_{fp.replace("/","_")}.txt'),'r',encoding='utf-8') as f: c=f.read()
                ws=set(re.findall(r'\b\w{3,}\b',c.lower())); m=sum(1 for w in q if w in ws)
                if m>0: res.append((fp,m/len(q)))
            except: continue
        res.sort(key=lambda x:x[1],reverse=True); return res[:limit]
    def clear_cache(self):
        for f in os.listdir(self.cache_dir):
            if f.endswith('.txt'): os.remove(os.path.join(self.cache_dir,f))
        self.repo_index=defaultdict(set)
    def save_cache(self):
        with open(os.path.join(self.cache_dir,'repo_index.json'),'w') as f:
            json.dump({k:list(v) for k,v in self.repo_index.items()},f)
    def load_cache(self):
        p=os.path.join(self.cache_dir,'repo_index.json')
        if os.path.exists(p):
            with open(p,'r') as f:
                try: ld=json.load(f); self.repo_index=defaultdict(set,{k:set(v) for k,v in ld.items()})
                except: self.repo_index=defaultdict(set)

