# prompt_editor.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import re, yaml

APP=FastAPI()
APP.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
ROOT=Path("agent/prompts")
TOK=lambda s:set(re.findall(r"\{([a-zA-Z0-9_]+)\}",s or ""))

REQ_SYS={
 "default_chat":{"amygdala_response"},
 "default_web_chat":{"amygdala_response"},
 "repo_file_chat":{"amygdala_response"},
 "channel_summarization":{"amygdala_response"},
 "ask_repo":{"amygdala_response"},
 "thought_generation":{"amygdala_response"},
 "file_analysis":{"amygdala_response"},
 "image_analysis":{"amygdala_response"},
 "combined_analysis":{"amygdala_response"},
 "attention_triggers":set()
}
REQ_FMT={
 "chat_with_memory":{"context","user_name","user_message"},
 "introduction":{"context","user_name","user_message"},
 "introduction_web":{"context","user_name","user_message"},
 "analyze_code":{"context","code_content","user_name","user_message"},
 "summarize_channel":{"channel_name","channel_history"},
 "ask_repo":{"context","question"},
 "repo_file_chat":{"file_path","code_type","repo_code","user_task_description","context"},
 "generate_thought":{"user_name","memory_text","timestamp","conversation_context"},
 "analyze_image":{"context","filename","user_message","user_name"},
 "analyze_file":{"context","filename","file_content","user_message","user_name"},
 "analyze_combined":{"context","image_files","text_files","user_message","user_name"}
}

class SaveBody(BaseModel):
    system_prompts: dict|None=None
    prompt_formats: dict|None=None

class CreateBody(BaseModel):
    name: str

def agents():
    if not ROOT.exists(): return []
    return sorted([p.name for p in ROOT.iterdir() if p.is_dir() and (p/"system_prompts.yaml").exists() and (p/"prompt_formats.yaml").exists()])

def load_yaml(p:Path):
    try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except: raise HTTPException(400,"invalid yaml")

def save_yaml(p:Path,data:dict):
    p.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding="utf-8")

def validate_map(d:dict,req:dict):
    out={}
    for k,v in d.items():
        if isinstance(v,str): found=TOK(v)
        elif isinstance(v,dict): found=set().union(*[TOK(x) for x in v.values() if isinstance(x,str)]) if v else set()
        elif isinstance(v,list): found=TOK("\n".join([x for x in v if isinstance(x,str)]))
        else: found=set()
        need=req.get(k,set())
        out[k]={"found":sorted(found),"required":sorted(need),"missing":sorted(need-found),"extra":sorted(found-need)}
    for k in set(req.keys())-set(d.keys()):
        if req[k]: out[k]={"found":[],"required":sorted(req[k]),"missing":sorted(req[k]),"extra":[]}
    return out

def sanitize_name(n:str):
    n=(n or "").strip()
    if not n: raise HTTPException(400,"empty name")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+",n): raise HTTPException(400,"invalid name")
    return n

def stub_system():
    return {
        "default_chat":       'you are blank. intensity {amygdala_response}%.\n{themes}\n',
        "default_web_chat":   'web face. intensity {amygdala_response}%.\n',
        "repo_file_chat":     'read code. intensity {amygdala_response}%.\n',
        "channel_summarization": 'summarize. intensity {amygdala_response}%.\n',
        "ask_repo":           'rag the repo. intensity {amygdala_response}%.\n',
        "thought_generation": 'reflect. intensity {amygdala_response}%.\n',
        "file_analysis":      'analyze text. intensity {amygdala_response}%.\n',
        "image_analysis":     'analyze image. intensity {amygdala_response}%.\n',
        "combined_analysis":  'analyze multimodal. intensity {amygdala_response}%.\n',
        "attention_triggers": []
    }

def stub_formats():
    return {
        "chat_with_memory": "{assembled_context}\n@{user_name}: {user_message}\n",
        "introduction": "{assembled_context}\n@{user_name}: {user_message}\n",
        "introduction_web": "{assembled_context}\n@{user_name}: {user_message}\n",
        "analyze_code": "{assembled_context}\n<code>\n{code_content}\n</code>\n@{user_name}: {user_message}\n",
        "summarize_channel": "{assembled_context}\n{content}\n",
        "ask_repo": "{assembled_context}\n{question}\n",
        "repo_file_chat": "{assembled_context}\nFile: {file_path}\nType: {code_type}\nContent:\n{repo_code}\nTask: {user_task_description}\n",
        "generate_thought": "Memory about @{user_name}: {memory_text}\nTimestamp: {timestamp}\n{conversation_context}\n",
        "analyze_image": "{assembled_context}\nImage: {filename}\n@{user_name}: {user_message}\n",
        "analyze_file": "{assembled_context}\nFile: {filename}\nContent:\n{file_content}\n@{user_name}: {user_message}\n",
        "analyze_combined": "{assembled_context}\nImages:\n{image_files}\nText:\n{text_files}\n@{user_name}: {user_message}\n"
    }

@APP.get("/agents")
def list_agents(): return {"agents":agents()}

@APP.post("/agents/new")
def create_agent(body:CreateBody):
    name=sanitize_name(body.name)
    p=ROOT/name
    if p.exists(): raise HTTPException(409,"agent exists")
    p.mkdir(parents=True,exist_ok=False)
    sys,fmt=stub_system(),stub_formats()
    save_yaml(p/"system_prompts.yaml",sys)
    save_yaml(p/"prompt_formats.yaml",fmt)
    return {"created":True,"name":name,"system_prompts":sys,"prompt_formats":fmt}

@APP.get("/agents/{name}")
def get_agent(name:str):
    p=ROOT/name
    if name not in agents(): raise HTTPException(404,"agent not found")
    sys=load_yaml(p/"system_prompts.yaml")
    fmt=load_yaml(p/"prompt_formats.yaml")
    return {"name":name,"system_prompts":sys,"prompt_formats":fmt,"states":{"system":sorted(sys.keys()),"formats":sorted(fmt.keys())}}

@APP.post("/agents/{name}")
def save_agent(name:str,body:SaveBody):
    p=ROOT/name
    if name not in agents(): raise HTTPException(404,"agent not found")
    if body.system_prompts is not None: save_yaml(p/"system_prompts.yaml",body.system_prompts)
    if body.prompt_formats is not None: save_yaml(p/"prompt_formats.yaml",body.prompt_formats)
    return {"ok":True}

@APP.get("/validate/{name}")
def validate_agent(name:str):
    p=ROOT/name
    if name not in agents(): raise HTTPException(404,"agent not found")
    sys=load_yaml(p/"system_prompts.yaml")
    fmt=load_yaml(p/"prompt_formats.yaml")
    sys_v=validate_map(sys,REQ_SYS)
    fmt_v=validate_map(fmt,REQ_FMT)
    avail={"system":sorted({t for k,v in sys_v.items() for t in v["found"]}),
           "formats":sorted({t for k,v in fmt_v.items() for t in v["found"]})}
    return {"name":name,"system":sys_v,"formats":fmt_v,"available_tokens":avail}

@APP.get("/",response_class=HTMLResponse)
def ui(_:Request):
    html="""
<!doctype html><meta charset="utf-8"/><title>Prompt Editor</title>
<style>
:root{--butt-pink:#ff9c9c;--butt-black:#000}
*{box-sizing:border-box}body{background:var(--butt-pink);color:var(--butt-black);font-family:"Courier New",monospace;margin:0}
.container{max-width:1200px;margin:0 auto;padding:2rem}
h1,h2{font-weight:400;text-transform:uppercase;margin:1.2em 0 .4em}
h1{font-size:2rem}h2{font-size:1.2rem}
select,textarea,button,input{background:var(--butt-pink);color:var(--butt-black);border:1px solid var(--butt-black);font-family:"Courier New",monospace;font-size:1rem;padding:.5rem}
select,button,input{width:auto}
textarea{width:100%;height:auto;overflow:hidden;resize:none}
button{cursor:pointer;text-transform:uppercase}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{border:1px solid var(--butt-black);padding:12px}
.badge{display:inline-block;border:1px solid var(--butt-black);padding:2px 6px;margin:2px;border-radius:10px}
.row{display:flex;gap:8px;align-items:center;margin:.5rem 0;flex-wrap:wrap}
.small{font-size:.9rem;opacity:.8}
.table{width:100%;border-collapse:collapse}
.table td,.table th{border:1px solid var(--butt-black);padding:6px;vertical-align:top}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
<div class="container">
<h1>Prompt Editor</h1>
<div class="row">
  <select id="agent"></select>
  <button id="load">Load</button>
  <button id="save">Save</button>
  <button id="validate">Validate</button>
  <input id="newName" placeholder="NEW agent name"/>
  <button id="newBtn">NEW</button>
</div>
<div id="states" class="row small"></div>
<div class="grid">
  <div class="card"><h2>System Prompts</h2><div id="sys"></div></div>
  <div class="card"><h2>Prompt Formats</h2><div id="fmt"></div></div>
</div>
<div class="card"><h2>Validation</h2><table class="table" id="val"><tr><th>Type</th><th>Key</th><th>Missing</th><th>Extra</th><th>Found</th><th>Provided</th></tr></table><div id="tokens" class="row small"></div></div>
</div>
<script>
const $=q=>document.querySelector(q);const C=(t,c)=>{let e=document.createElement(t);if(c)e.className=c;return e}
function autosize(t){t.style.height='auto';t.style.height=(t.scrollHeight)+'px'}
function autosizeAll(root){root.querySelectorAll('textarea').forEach(autosize)}
async function list(){let r=await fetch('/agents');let j=await r.json();let s=$("#agent");s.innerHTML='';j.agents.forEach(a=>{let o=document.createElement('option');o.value=a;o.textContent=a;s.appendChild(o)})}
function render(obj,root){
 root.innerHTML='';
 Object.entries(obj).forEach(([k,v])=>{
  let l=C('div');l.textContent=k;
  let ta=C('textarea');ta.value=typeof v==="string"?v:JSON.stringify(v,null,2);ta.dataset.key=k;
  ta.addEventListener('input',e=>autosize(e.target));
  root.appendChild(l);root.appendChild(ta);autosize(ta)
 })
}
async function load(){
 let a=$("#agent").value;if(!a)return;let r=await fetch('/agents/'+a);let j=await r.json();
 window._agent=a;window._sys=j.system_prompts;window._fmt=j.prompt_formats;
 $("#states").textContent='system: '+j.states.system.join(', ')+' | formats: '+j.states.formats.join(', ');
 render(window._sys,$("#sys"));render(window._fmt,$("#fmt"))
}
async function save(){
 let a=window._agent;if(!a)return;
 let sys={},fmt={};
 $("#sys").querySelectorAll('textarea').forEach(t=>{let k=t.dataset.key;let v=t.value;try{sys[k]=/^[\\[{]/.test(v.trim())?JSON.parse(v):v}catch{sys[k]=v}});
 $("#fmt").querySelectorAll('textarea').forEach(t=>{let k=t.dataset.key;let v=t.value;try{fmt[k]=/^[\\[{]/.test(v.trim())?JSON.parse(v):v}catch{fmt[k]=v}});
 let r=await fetch('/agents/'+a,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({system_prompts:sys,prompt_formats:fmt})});let j=await r.json();if(j.ok)await load()
}
function cell(x){let d=C('td');if(Array.isArray(x))d.innerHTML=x.map(y=>'<span class="badge">'+y+'</span>').join(' ');else d.textContent=x;return d}
async function validate(){
 let a=window._agent;if(!a)return;let r=await fetch('/validate/'+a);let j=await r.json();let T=$("#val");
 T.innerHTML='<tr><th>Type</th><th>Key</th><th>Missing</th><th>Extra</th><th>Found</th><th>Provided</th></tr>';
 for(let [k,v] of Object.entries(j.system)){let tr=C('tr');tr.appendChild(cell('system'));tr.appendChild(cell(k));tr.appendChild(cell(v.missing));tr.appendChild(cell(v.extra));tr.appendChild(cell(v.found));tr.appendChild(cell(v.required));T.appendChild(tr)}
 for(let [k,v] of Object.entries(j.formats)){let tr=C('tr');tr.appendChild(cell('format'));tr.appendChild(cell(k));tr.appendChild(cell(v.missing));tr.appendChild(cell(v.extra));tr.appendChild(cell(v.found));tr.appendChild(cell(v.required));T.appendChild(tr)}
 $("#tokens").textContent='available tokens — system: '+j.available_tokens.system.join(', ')+' | formats: '+j.available_tokens.formats.join(', ')
}
async function createNew(){
 let n=$("#newName").value.trim();if(!n){alert('name?');return}
 let r=await fetch('/agents/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});
 if(!r.ok){let e=await r.text();alert('error: '+e);return}
 await list();$("#agent").value=n;await load()
}
window.addEventListener('resize',()=>{autosizeAll(document)})
$("#load").onclick=load;$("#save").onclick=save;$("#validate").onclick=validate;$("#newBtn").onclick=createNew;list()
</script>
"""
    return HTMLResponse(html)

if __name__=="__main__":
    import uvicorn; uvicorn.run(APP,host="127.0.0.1",port=8005)
