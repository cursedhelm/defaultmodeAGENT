from typing import List, Tuple, Dict
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import re, logging, threading, itertools, os, json, pickle, time
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from bot_config import config

logger=logging.getLogger(__name__)


class ThemePrompts(BaseModel):
    """Formats for the theme block substituted into the YAML {themes} placeholder.

    This is the memory index whispering its n-gram statistics into the persona
    prompt. Unknown modes fall through to the bullet format.
    """
    tagged_user: str = Field(default="User:{theme}")
    tagged_bot: str = Field(default="Bot:{theme}")
    sections: str = Field(default="Current User Preferences:\n{user_themes}\n\nYour Global Preferences:\n{global_themes}")
    just_user: str = Field(default="Current User Preferences:\n{user_themes}")
    just_global: str = Field(default="Your Global Preferences:\n{global_themes}")
    bullet_user: str = Field(default="- U:{theme}")
    bullet_bot: str = Field(default="- G:{theme}")


PROMPTS = ThemePrompts()

_TRIGRAM_CACHE:List[str]=[]
_CACHE_EXPIRES:datetime=datetime.min.replace(tzinfo=timezone.utc)
_REFRESHING=False
_LOCK=threading.Lock()
_USER_THEME_CACHE:Dict[str,List[str]]=defaultdict(list)
_USER_EXPIRES:Dict[str,datetime]=defaultdict(lambda:datetime.min.replace(tzinfo=timezone.utc))
_USER_REFRESHING:Dict[str,bool]=defaultdict(bool)
_LAST_TRIGGER_TIME:datetime=datetime.min.replace(tzinfo=timezone.utc)
_LAST_TRIGGER_TIME_BY_USER:Dict[str,datetime]=defaultdict(lambda:datetime.min.replace(tzinfo=timezone.utc))


def snapshot_theme_cache()->Dict:
    """Return current in-memory themes without loading, refreshing, or saving."""
    with _LOCK:
        global_themes=list(_TRIGRAM_CACHE)
        # User writers replace whole lists, so shallow copies preserve a stable
        # read-only transport snapshot without invoking theme extraction.
        user_items=list(_USER_THEME_CACHE.items())
        global_expires=_CACHE_EXPIRES.timestamp() if _CACHE_EXPIRES.year>1 else None
    return {
        'global_themes':global_themes,
        'user_themes':{str(uid):list(themes) for uid,themes in user_items},
        'expires_at_epoch':global_expires,
    }

def format_themes_for_prompt(mi,uid:str,spike:bool=False,k_user:int=12,k_global:int=8,mode:str="just_user")->str:
    ut=get_user_themes(mi,uid) if uid else []
    gt=get_current_themes(mi)
    if spike:k_user*=2
    ut=ut[:k_user]
    gt=[t for t in gt if t not in ut][:k_global]
    if mode=="inline":return ", ".join(ut+gt)
    if mode=="tagged":return ", ".join([PROMPTS.tagged_user.format(theme=t) for t in ut]+[PROMPTS.tagged_bot.format(theme=t) for t in gt])
    if mode=="sections":return PROMPTS.sections.format(user_themes=", ".join(ut),global_themes=", ".join(gt))
    if mode=="just_user":return PROMPTS.just_user.format(user_themes=", ".join(ut))
    if mode=="just_global":return PROMPTS.just_global.format(global_themes=", ".join(gt))
    return "\n".join([*(PROMPTS.bullet_user.format(theme=t) for t in ut),*(PROMPTS.bullet_bot.format(theme=t) for t in gt)])


def _tok(x:str)->List[str]:
    return re.findall(r"\b[a-z0-9]{3,}\b",x.lower())

def _ng(t:List[str],n:int=3)->List[Tuple[str,...]]:
    return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]

def _sg(t:List[str],n:int=3,max_gap:int=2)->List[Tuple[str,...]]:
    L=len(t)
    if L<n:return []
    out=[];w=n+max_gap
    for i in range(L):
        win=t[i:i+w]
        if len(win)<n:break
        for idxs in itertools.combinations(range(len(win)),n):
            if all((idxs[k+1]-idxs[k])<=1+max_gap for k in range(n-1)):
                out.append(tuple(win[j] for j in idxs))
    return out

def _texts_global(mi)->List[str]:
    return [m for m in getattr(mi,'memories',[]) if m is not None]

def _texts_user(mi,user_id:str)->List[str]:
    if not hasattr(mi,'user_memories') or not hasattr(mi,'memories'):return []
    ids=mi.user_memories.get(user_id,[])
    mem=mi.memories
    return [mem[i] for i in ids if i<len(mem) and mem[i] is not None]

def _get_user_count(mi)->int:
    return len(getattr(mi,'user_memories',{}))

def _get_memory_stats(mi)->Dict:
    mem=getattr(mi,'memories',[])
    act=sum(1 for m in mem if m is not None)
    u=_get_user_count(mi)
    return {'total_memories':len(mem),'active_memories':act,'users':u,'avg_memories_per_user':(act/u if u>0 else 0)}

def _extract(texts:List[str],top_n=None,min_occ=None,allow_repeated_tokens:bool=False)->List[str]:
    if top_n is None: top_n=config.attention.default_top_n
    if min_occ is None: min_occ=config.attention.default_min_occ
    c=Counter()
    for x in texts:
        if not x:continue
        toks=_tok(x)
        c.update(_ng(toks,3));c.update(_sg(toks,3,2))
    filt=[]
    sw=config.attention.stop_words
    for tg,count in c.items():
        if count<min_occ:continue
        if any(w in sw for w in tg):continue
        if not allow_repeated_tokens and len(set(tg))<3:continue
        filt.append(tg)
    filt.sort(key=lambda tg:(c[tg],len(set(tg))," ".join(tg)),reverse=True)
    seen=set();out=[]
    for tg in filt:
        w=list(tg)
        if any(x in seen for x in w):continue
        out.append(" ".join(w));seen.update(w)
        if len(out)>=top_n:break
    return out

def _base_dir(mi):
    base=os.path.join('cache','default','themes')
    cm=getattr(mi,'cache_manager',None)
    if cm and hasattr(cm,'get_cache_dir'): base=cm.get_cache_dir('themes')
    os.makedirs(base,exist_ok=True);return base

def _theme_paths(mi):
    b=_base_dir(mi);return os.path.join(b,'themes.pkl'),os.path.join(b,'themes.meta.json')

def _user_paths(mi,uid:str):
    b=os.path.join(_base_dir(mi),'users',uid);os.makedirs(b,exist_ok=True)
    return os.path.join(b,'themes.pkl'),os.path.join(b,'themes.meta.json')

def _load_global(mi)->bool:
    global _TRIGRAM_CACHE,_CACHE_EXPIRES
    cp,mp=_theme_paths(mi)
    if not (os.path.exists(cp) and os.path.exists(mp)):return False
    try:
        with open(cp,'rb') as f: th=pickle.load(f)
        with open(mp,'r',encoding='utf-8') as m: meta=json.load(m)
        th=list(th or []);upd=float(meta.get('updated_at_epoch',0))
        with _LOCK:
            _TRIGRAM_CACHE=th
            _CACHE_EXPIRES=(datetime.fromtimestamp(upd, tz=timezone.utc)+config.attention.refresh_interval if upd>0 else datetime.min.replace(tzinfo=timezone.utc))
        logger.info(f"Loaded global themes: {len(_TRIGRAM_CACHE)}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load global themes: {e}");return False

def _save_global(mi,themes:list):
    global _TRIGRAM_CACHE,_CACHE_EXPIRES
    cp,mp=_theme_paths(mi);tmp=cp+'.tmp'
    with open(tmp,'wb') as f: pickle.dump(list(themes),f)
    os.replace(tmp,cp)
    with open(mp,'w',encoding='utf-8') as m: json.dump({'updated_at_epoch':time.time(),'count':len(themes)},m)
    old=set(_TRIGRAM_CACHE);new=set(themes)
    with _LOCK:
        _TRIGRAM_CACHE=list(themes)
        _CACHE_EXPIRES=datetime.now(timezone.utc)+config.attention.refresh_interval
    logger.info(f"GLOBAL THEMES REFRESHED: {len(themes)}")
    logger.info(f"Next refresh: {_CACHE_EXPIRES.strftime('%H:%M:%S')}")
    if themes: logger.info(f"Top themes: {themes}")
    added=new-old;removed=old-new
    if added: logger.info(f"New themes: {list(added)}")
    if removed: logger.info(f"Removed themes: {list(removed)}")
    if not added and not removed and old: logger.info("Themes unchanged")

def _load_user(mi,uid:str)->bool:
    cp,mp=_user_paths(mi,uid)
    if not (os.path.exists(cp) and os.path.exists(mp)):return False
    try:
        with open(cp,'rb') as f: th=pickle.load(f)
        with open(mp,'r',encoding='utf-8') as m: meta=json.load(m)
        _USER_THEME_CACHE[uid]=list(th or [])
        upd=float(meta.get('updated_at_epoch',0))
        _USER_EXPIRES[uid]=(datetime.fromtimestamp(upd, tz=timezone.utc)+config.attention.refresh_interval if upd>0 else datetime.min.replace(tzinfo=timezone.utc))
        logger.info(f"Loaded user themes for {uid}: {len(_USER_THEME_CACHE[uid])}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load user themes for {uid}: {e}");return False

def _save_user(mi,uid:str,themes:list):
    cp,mp=_user_paths(mi,uid);tmp=cp+'.tmp'
    with open(tmp,'wb') as f: pickle.dump(list(themes),f)
    os.replace(tmp,cp)
    with open(mp,'w',encoding='utf-8') as m: json.dump({'updated_at_epoch':time.time(),'count':len(themes)},m)
    old=set(_USER_THEME_CACHE[uid]);new=set(themes)
    _USER_THEME_CACHE[uid]=list(themes)
    _USER_EXPIRES[uid]=datetime.now(timezone.utc)+config.attention.refresh_interval
    logger.info(f"USER THEMES REFRESHED [{uid}]: {len(themes)}")
    logger.info(f"Next refresh: {_USER_EXPIRES[uid].strftime('%H:%M:%S')}")
    if themes: logger.info(f"Top themes: {themes[:3]}")
    added=new-old;removed=old-new
    if added: logger.info(f"New themes: {list(added)[:2]}")
    if removed: logger.info(f"Removed themes: {list(removed)[:2]}")
    if not added and not removed and old: logger.info("Themes unchanged")

def _maybe_refresh_global(mi):
    global _REFRESHING
    if datetime.now(timezone.utc)<_CACHE_EXPIRES:return
    with _LOCK:
        if _REFRESHING:
            logger.debug("Global refresh in progress");return
        _REFRESHING=True
    logger.info("Starting background global theme refresh...")
    def worker():
        global _REFRESHING
        try:
            t0=time.time();stats=_get_memory_stats(mi)
            logger.info(f"Processing {stats['active_memories']} memories from {stats['users']} users")
            _save_global(mi,_extract(_texts_global(mi)))
            logger.info(f"Global theme refresh completed in {time.time()-t0:.2f}s")
        except Exception as e:
            logger.error(f"Global theme refresh failed: {e}")
        finally:
            with _LOCK:_REFRESHING=False
    threading.Thread(target=worker,daemon=True).start()

def _maybe_refresh_user(mi,uid:str):
    if datetime.now(timezone.utc)<_USER_EXPIRES[uid]:return
    if _USER_REFRESHING[uid]:
        logger.debug(f"User refresh for {uid} in progress");return
    _USER_REFRESHING[uid]=True
    logger.info(f"Starting background user theme refresh for {uid}...")
    def worker():
        try:
            t0=time.time();ut=_texts_user(mi,uid)
            logger.info(f"Processing {len(ut)} memories for user {uid}")
            _save_user(mi,uid,_extract(ut))
            logger.info(f"User theme refresh for {uid} completed in {time.time()-t0:.2f}s")
        except Exception as e:
            logger.error(f"User theme refresh for {uid} failed: {e}")
        finally:
            _USER_REFRESHING[uid]=False
    threading.Thread(target=worker,daemon=True).start()

def _dynamic_triggers(base:List[str],mi=None,uid:str=None)->List[str]:
    s=set(base)
    if mi:s.update(_TRIGRAM_CACHE)
    if mi and uid:s.update(get_user_themes(mi,uid))
    return list(s)

def check_attention_triggers_fuzzy(content:str,persona_triggers:List[str],threshold:int=None,*,memory_index=None,top_n=None,cooldown=None,user_id:str=None)->bool:
    global _LAST_TRIGGER_TIME,_LAST_TRIGGER_TIME_BY_USER
    if threshold is None: threshold=config.attention.threshold
    if top_n is None: top_n=config.attention.default_top_n
    if cooldown is None: cooldown=config.attention.cooldown
    if not content:return False
    now=datetime.now(timezone.utc)
    if user_id:
        if now-_LAST_TRIGGER_TIME_BY_USER[user_id]<cooldown:return False
    else:
        if now-_LAST_TRIGGER_TIME<cooldown:return False
    if memory_index:_maybe_refresh_global(memory_index)
    triggers=_dynamic_triggers(persona_triggers,memory_index,user_id)
    cl=content.lower().strip();words=cl.split()
    if len(words)<4:return False
    for trig in triggers:
        if not trig:continue
        t=trig.lower().strip()
        if t in cl:
            if user_id:_LAST_TRIGGER_TIME_BY_USER[user_id]=now
            else:_LAST_TRIGGER_TIME=now
            return True
        tw=[w for w in t.split() if len(w)>=3]
        if not tw:continue
        scores=[max((fuzz.ratio(cw,twk) for cw in words),default=0) for twk in tw]
        if scores:
            cov=sum(1 for sc in scores if sc>=threshold)/len(tw)
            avg=sum(scores)/len(scores)
            mincov=0.5+0.5/len(tw)
            if cov>=mincov and avg>=threshold:
                if user_id:_LAST_TRIGGER_TIME_BY_USER[user_id]=now
                else:_LAST_TRIGGER_TIME=now
                return True
    return False

def get_current_themes(memory_index=None)->List[str]:
    if memory_index:
        if not _TRIGRAM_CACHE:
            if not _load_global(memory_index):
                logger.info("Initializing global themes from memory corpus...")
                _save_global(memory_index,_extract(_texts_global(memory_index),min_occ=2))
            else:_maybe_refresh_global(memory_index)
        else:_maybe_refresh_global(memory_index)
    return list(_TRIGRAM_CACHE)

def get_user_themes(memory_index,user_id:str)->List[str]:
    if not _USER_THEME_CACHE[user_id]:
        if not _load_user(memory_index,user_id):
            logger.info(f"Initializing themes for user {user_id}...")
            _save_user(memory_index,user_id,_extract(_texts_user(memory_index,user_id),min_occ=2))
        else:_maybe_refresh_user(memory_index,user_id)
    else:_maybe_refresh_user(memory_index,user_id)
    return list(_USER_THEME_CACHE[user_id])

def get_themes(memory_index,user_id:str=None,include_global:bool=True)->List[str]:
    g=get_current_themes(memory_index) if include_global else []
    u=get_user_themes(memory_index,user_id) if user_id else []
    return list(dict.fromkeys([*u,*g]))

def active_triggers(persona_triggers:List[str],memory_index=None,user_id:str=None)->List[str]:
    if memory_index:_maybe_refresh_global(memory_index)
    return list(dict.fromkeys(_dynamic_triggers(persona_triggers,memory_index,user_id)))

def warm_theme_cache(memory_index)->int:
    """Populate the global theme cache ahead of first use.

    Safe to call redundantly and safe to skip entirely — get_current_themes
    self-initializes. This only moves the cold-start cost (full trigram
    extraction over the corpus, when no pickle exists) off the first request
    and onto startup. Synchronous and blocking; callers on an event loop
    should dispatch it via asyncio.to_thread. Returns the theme count.
    """
    try:
        return len(get_current_themes(memory_index))
    except Exception as e:
        logger.warning(f"Theme cache warm failed: {e}")
        return 0

def force_rebuild_theme_cache(memory_index)->List[str]:
    th=_extract(_texts_global(memory_index));_save_global(memory_index,th);return th

def force_rebuild_user_theme_cache(memory_index,user_id:str)->List[str]:
    th=_extract(_texts_user(memory_index,user_id));_save_user(memory_index,user_id,th);return th

def invalidate_global_cache():
    global _TRIGRAM_CACHE,_CACHE_EXPIRES,_REFRESHING
    _TRIGRAM_CACHE=[];_CACHE_EXPIRES=datetime.min.replace(tzinfo=timezone.utc);_REFRESHING=False

def invalidate_user_cache(user_id:str):
    _USER_THEME_CACHE.pop(user_id,None);_USER_EXPIRES[user_id]=datetime.min.replace(tzinfo=timezone.utc);_USER_REFRESHING[user_id]=False

def purge_theme_cache(memory_index,user_id:str=None):
    if user_id is None:
        cp,mp=_theme_paths(memory_index)
        for p in (cp,mp):
            if os.path.exists(p):os.remove(p)
        invalidate_global_cache()
    else:
        cp,mp=_user_paths(memory_index,user_id)
        for p in (cp,mp):
            if os.path.exists(p):os.remove(p)
        invalidate_user_cache(user_id)

def theme_cache_info(memory_index=None,user_id:str=None)->Dict:
    info={'global_themes':{'count':len(_TRIGRAM_CACHE),'expires':_CACHE_EXPIRES.isoformat(),'refreshing':_REFRESHING}}
    if user_id:
        info['user_themes']={'count':len(_USER_THEME_CACHE[user_id]),'expires':_USER_EXPIRES[user_id].isoformat(),'refreshing':_USER_REFRESHING[user_id]}
    if memory_index:
        info['memory_system']=_get_memory_stats(memory_index)
        if user_id and hasattr(memory_index,'user_memories'):
            ids=memory_index.user_memories.get(user_id,[])
            info['user_memory_stats']={'memory_count':len(ids),'valid_memories':len([i for i in ids if i<len(memory_index.memories) and memory_index.memories[i] is not None])}
    return info
