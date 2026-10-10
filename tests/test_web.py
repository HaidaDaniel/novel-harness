"""Offline browser-script contract for the book-first wizard."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_wizard_dom_and_api_transitions_without_model_or_worker():
    if shutil.which("node") is None:
        pytest.skip("Node is unavailable for the static browser-script fixture")
    html = (ROOT / "src/novel_harness/web/index.html").read_text("utf-8")
    for element_id in (
        "wizard-novel", "wizard-preset", "wizard-profile", "wizard-execution",
        "wizard-start-sample", "wizard-approve", "wizard-whole-book",
        "raw-create-controls", "runs-table", "review-desk", "btn-pause",
    ):
        assert f'id="{element_id}"' in html

    script = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: '', textContent: '', innerHTML: '', disabled: false, open: false,
    style: {}, dataset: {}, listeners: {},
    classList: { toggle() {} },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    querySelector() { return node('selector-child'); },
    querySelectorAll() { return []; },
    scrollIntoView() {},
  });
  return nodes.get(id);
}
const document = {
  getElementById: node,
  querySelector() { return node('runs-body'); },
  querySelectorAll() { return []; },
};
const calls = [];
const approvals = {};
const statuses = {};
let sampleCount = 0;
let wholeCount = 0;
let failStart = false;
let novels = [
  {novel_id:'book-a',title:'Book A',status:'ready',chapter_count:2,translation_state:'new'},
  {novel_id:'book-b',title:'Book B',status:'ready',chapter_count:2,translation_state:'new'},
];
const profiles = [
  {id:'oc-inherit', available:true, configured_profile:true, harness:'opencode',
   model_policy:'inherit', model_source:'inherited from OpenCode agent', agent_id:'translator',
   requested_model:{policy:'inherit'}, resolved_model:{provider:'private',model:'model-a'}},
  {id:'oc-override', available:true, configured_profile:true, harness:'opencode',
   model_policy:'override', model_source:'explicit advanced override', agent_id:'translator',
   requested_model:{id:'private/model-b'}, resolved_model:{provider:'private',model:'model-b'}},
  {id:'unavailable', available:false, configured_profile:true, harness:'opencode',
   reason:'unknown agent', resolved_model:{status:'unknown'}},
];
const modes = [
  {id:'quality',label:'Quality',available:true,pipeline:'direct',
   execution_profile_id:'oc-inherit',stages:[{name:'translate',profile_id:'oc-inherit'}]},
  {id:'broken',label:'Broken',available:false,reason:'connection unavailable',
   execution_profile_id:'unavailable',pipeline:'direct',stages:[]},
];
function response(body, status=200) {
  return {ok:status<400,status,statusText:'failure',text:async()=>JSON.stringify(body)};
}
async function fetch(path, options={}) {
  const body = options.body ? JSON.parse(options.body) : null;
  calls.push({path,method:options.method||'GET',body});
  if (path === '/wizard/options') return response({presets:modes,default:'quality',profiles});
  if (path === '/presets') return response({presets:[],default:null});
  if (path === '/runners') return response([]);
  if (path === '/library') return response({novels,diagnostics:[]});
  if (path === '/runs') return response({runs:[]});
  if (path === '/sample-runs' && options.method === 'POST') {
    const runId = `sample-${++sampleCount}`;
    statuses[runId] = {status:'created',generation_complete:false};
    return response({run_id:runId},201);
  }
  if (path === '/whole-book-runs' && options.method === 'POST') {
    if (!approvals[body.sample_run_id]) return response({detail:'approval required'},409);
    const runId = `whole-${++wholeCount}`;
    statuses[runId] = {status:'running',generation_complete:false};
    return response({run_id:runId},201);
  }
  const sampleApproval = path.match(/^\/sample-runs\/(sample-\d+)\/approval$/);
  if (sampleApproval) return response(approvals[sampleApproval[1]] || null);
  const approve = path.match(/^\/sample-runs\/(sample-\d+)\/approve$/);
  if (approve) {
    if (statuses[approve[1]].status !== 'completed') return response({detail:'incomplete'},400);
    approvals[approve[1]] = {approved_at:'now'};
    return response(approvals[approve[1]]);
  }
  const runPath = path.match(/^\/runs\/(sample-\d+|whole-\d+)\/(start|status|progress|metrics)$/);
  if (runPath) {
    const [,runId,action] = runPath;
    if (action === 'start') {
      if (failStart) { failStart = false; return response({detail:'worker unavailable'},400); }
      statuses[runId] = {status:'running',generation_complete:false};
      return response({started:true});
    }
    if (action === 'progress') return response({current:null});
    if (action === 'metrics') return response({stages:[]});
    return response({run_id:runId, ...statuses[runId], runners:[], chapters:[], chunks:{},
      workflow:'direct', progress:0, total_chunks:0, created_at:'now', updated_at:'now'});
  }
  throw new Error(`Unexpected request: ${path}`);
}
const context = vm.createContext({document,fetch,setInterval:()=>1,clearInterval:()=>{}});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const call = (id, event) => node(id).listeners[event]();
(async () => {
  await new Promise(setImmediate);
  assert.equal(node('wizard-preset').value,'quality');
  assert.equal(node('wizard-profile').value,'oc-inherit');
  assert(node('wizard-profile-note').textContent.includes('model-a'));
  assert(node('wizard-profile-note').textContent.includes('inherited from OpenCode agent'));
  assert.equal(node('wizard-start-sample').disabled,true); // no book selected
  node('wizard-preset').value='broken'; call('wizard-preset','change');
  node('wizard-novel').value='book-a'; call('wizard-novel','change');
  assert.equal(node('wizard-start-sample').disabled,true); // unavailable preset/profile
  node('wizard-preset').value='quality'; call('wizard-preset','change');
  node('wizard-novel').value='book-a'; call('wizard-novel','change');
  assert.equal(node('wizard-start-sample').disabled,false);
  await Promise.all([call('wizard-start-sample','click'),call('wizard-start-sample','click')]);
  assert.equal(calls.filter(item=>item.path==='/sample-runs').length,1);
  assert.deepEqual(calls.find(item=>item.path==='/sample-runs').body,
    {novel_id:'book-a',preset:'quality'});
  assert.equal(node('wizard-approve').disabled,true);
  await call('wizard-approve','click');
  assert.equal(calls.filter(item=>item.path.endsWith('/approve')).length,0);
  statuses['sample-1']={status:'completed',generation_complete:true};
  await vm.runInContext('refreshWizardSample()',context);
  assert.equal(node('wizard-approve').disabled,false);
  await call('wizard-approve','click');
  assert.equal(node('wizard-whole-book').disabled,false);
  node('wizard-novel').value='book-b'; call('wizard-novel','change');
  assert.equal(node('wizard-whole-book').disabled,true);
  await call('wizard-whole-book','click');
  assert.equal(calls.filter(item=>item.path==='/whole-book-runs').length,0);
  node('wizard-novel').value='book-a'; call('wizard-novel','change');
  await Promise.all([call('wizard-whole-book','click'),call('wizard-whole-book','click')]);
  const whole = calls.filter(item=>item.path==='/whole-book-runs');
  assert.equal(whole.length,1);
  assert.deepEqual(whole[0].body,
    {sample_run_id:'sample-1',approved:true,novel_id:'book-a',preset:'quality'});
  node('wizard-novel').value='book-b'; call('wizard-novel','change');
  await call('wizard-start-sample','click');
  assert.deepEqual(calls.filter(item=>item.path==='/sample-runs')[1].body,
    {novel_id:'book-b',preset:'quality'});
  node('wizard-preset').value='custom'; call('wizard-preset','change');
  node('wizard-profile').value='oc-override'; call('wizard-profile','change');
  assert(node('wizard-execution').innerHTML.includes('explicit configured override'));
  failStart=true;
  await call('wizard-start-sample','click');
  assert.deepEqual(calls.filter(item=>item.path==='/sample-runs')[2].body,
    {novel_id:'book-b',execution_profile:'oc-override',pipeline:'direct'});
  assert(node('wizard-status').textContent.includes('worker did not start'));
  assert.equal(node('wizard-open-sample').disabled,false);
  assert.equal(calls.filter(item=>item.path==='/whole-book-runs').length,1);
  novels=[];
  await vm.runInContext('loadLibrary()',context);
  assert.equal(node('wizard-start-sample').disabled,true);
  assert(node('novel-cards').innerHTML.includes('No source novels found'));
  console.log(JSON.stringify({samples:sampleCount,whole:wholeCount}));
})().catch(error=>{ console.error(error); process.exitCode=1; });
"""
    result = subprocess.run(
        ["node", "-e", script, str(ROOT / "src/novel_harness/web/app.js")],
        check=False, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"samples": 3, "whole": 1}


def test_deferred_run_and_chapter_responses_cannot_replace_selection():
    if shutil.which("node") is None:
        pytest.skip("Node is unavailable for the static browser-script fixture")
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const src=fs.readFileSync(process.argv[1],'utf8');
const cut=(a,b)=>src.slice(src.indexOf(a),src.indexOf(b));

const runCtx=vm.createContext({});
vm.runInContext(
  "const state={runId:'A',runGeneration:0};const rendered=[];const pending={};" +
  "function api(p){return p.endsWith('/status')?" +
  "new Promise(r=>pending[p]=r):Promise.resolve(null);}" +
  "function renderRunSummary(s){rendered.push(s.run_id);}function renderRunProgress(){}" +
  "function renderRunUsage(){}function renderRunChapters(){}function setNotice(){}", runCtx);
vm.runInContext(cut("async function renderRun()","function renderRunSummary"), runCtx);

const chapterCtx=vm.createContext({});
vm.runInContext(
  "const state={runId:'A',runGeneration:0,chapter:null,chapterGeneration:0};const rendered=[];const pending={};" +
  "function el(id){return {value:'1'};}function api(p){return new Promise(r=>pending[p]=r);}" +
  "function clearChapterView(){}function renderChapter(d){rendered.push(d.run_id+':'+d.chapter);}function setNotice(){}", chapterCtx);
vm.runInContext(cut("function isCurrentChapter(","function renderChapter"), chapterCtx);

const reviewCtx=vm.createContext({});
vm.runInContext(
  "const state={runId:'A',runGeneration:0,chapter:1,chapterGeneration:0};let loadCalls=0;const pending={};" +
  "function api(p){return new Promise(r=>pending[p]=r);}function setNotice(){}" +
  "function loadChapter(){loadCalls++;}", reviewCtx);
vm.runInContext(cut("function isCurrentChapter(","async function loadChapter()"), reviewCtx);
vm.runInContext(cut("async function reviewAction(", "el(\"btn-load-chapter\")"), reviewCtx);

const switchCtx=vm.createContext({});
vm.runInContext(
  "const nodes={};function el(id){if(!nodes[id])nodes[id]=" +
  "{textContent:'old',innerHTML:'old',value:'5'};return nodes[id];}" +
  "function showTab(){}function renderRun(){}function clearInterval(){}" +
  "function setInterval(){return 1;}" +
  "const state={runId:'A',runGeneration:0,chapter:5,poll:null};", switchCtx);
vm.runInContext(cut("function clearChapterView(", "async function renderRun()"), switchCtx);

(async()=>{
  // openRun clears the previous chapter and review desk when the run switches.
  vm.runInContext("openRun('B')",switchCtx);
  assert.equal(vm.runInContext("state.chapter",switchCtx),null);
  assert.equal(vm.runInContext("state.runGeneration",switchCtx),1);
  assert.equal(vm.runInContext("el('review-desk').innerHTML",switchCtx),"");
  assert.equal(vm.runInContext("el('chapter-source').textContent",switchCtx),"");
  assert.equal(vm.runInContext("el('chapter-number').value",switchCtx),"");

  // A run switch discards the older run's late response.
  const a=vm.runInContext("renderRun()",runCtx);
  vm.runInContext("state.runId='B';state.runGeneration+=1",runCtx);
  const b=vm.runInContext("renderRun()",runCtx);
  vm.runInContext("pending['/runs/B/status']({run_id:'B'})",runCtx); await b;
  vm.runInContext("pending['/runs/A/status']({run_id:'A'})",runCtx); await a;
  assert.deepEqual(vm.runInContext("JSON.parse(JSON.stringify(rendered))",runCtx),["B"]);

  // A chapter load for the previous run cannot render into the new selection.
  const ca=vm.runInContext("loadChapter()",chapterCtx);
  vm.runInContext("state.runId='B';state.runGeneration+=1;state.chapter=null",chapterCtx);
  vm.runInContext("pending['/runs/A/chapters/1']({run_id:'A',chapter:1})",chapterCtx); await ca;
  assert.deepEqual(vm.runInContext("JSON.parse(JSON.stringify(rendered))",chapterCtx),[]);
  const cb=vm.runInContext("loadChapter()",chapterCtx);
  vm.runInContext("pending['/runs/B/chapters/1']({run_id:'B',chapter:1})",chapterCtx); await cb;
  assert.deepEqual(vm.runInContext("JSON.parse(JSON.stringify(rendered))",chapterCtx),["B:1"]);

  // A review action keeps its captured run/chapter and does not refresh a newer view.
  const ra=vm.runInContext(
    "reviewAction('accept','c1','machine-hash',null,{value:'x'},{runId:state.runId,chapter:state.chapter,generation:state.runGeneration,chapterGeneration:0})",reviewCtx);
  vm.runInContext("state.runId='B';state.runGeneration+=1;state.chapter=2",reviewCtx);
  const staleKey=vm.runInContext("Object.keys(pending).find(k=>k.includes('/A/'))",reviewCtx);
  vm.runInContext("pending["+JSON.stringify(staleKey)+"]({})",reviewCtx); await ra;
  assert.equal(vm.runInContext("loadCalls",reviewCtx),0);

  const rb=vm.runInContext(
    "reviewAction('accept','c1','machine-hash',null,{value:'x'},{runId:state.runId,chapter:state.chapter,generation:state.runGeneration,chapterGeneration:0})",reviewCtx);
  const liveKey=vm.runInContext("Object.keys(pending).find(k=>k.includes('/B/'))",reviewCtx);
  vm.runInContext("pending["+JSON.stringify(liveKey)+"]({})",reviewCtx); await rb;
  assert.equal(vm.runInContext("loadCalls",reviewCtx),1);

  console.log(JSON.stringify({ok:true}));
})().catch(error=>{ console.error(error); process.exitCode=1; });
"""
    result = subprocess.run(
        ["node", "-e", script, str(ROOT / "src/novel_harness/web/app.js")],
        check=False, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}


def test_chapter_controls_and_overlapping_requests_keep_the_rendered_identity():
    import json
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("Node is unavailable for the browser-script fixture")
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const src=fs.readFileSync(process.argv[1],'utf8');
const cut=(a,b)=>src.slice(src.indexOf(a),src.indexOf(b));
const ctx=vm.createContext({});
vm.runInContext(`
const state={runId:'A',runGeneration:0,chapter:1,chapterGeneration:0};
const nodes={};const pending=[];const writes=[];const rendered=[];let callback;
function el(id){return nodes[id]||(nodes[id]={value:'',textContent:'',innerHTML:''});}
const textarea={value:'old chapter text'};
const button={dataset:{act:'edit'},addEventListener:(_event,fn)=>{callback=fn;}};
const item={dataset:{chunk:'shared-chunk',machine:'shared-machine',finalSha:''},
 querySelector:()=>textarea,querySelectorAll:()=>[button]};
const document={querySelectorAll:()=>[item]};
function api(path,options){
 if(options){writes.push({path,body:JSON.parse(options.body)});return Promise.resolve({});}
 return new Promise((resolve,reject)=>pending.push({path,resolve,reject}));
}
function renderChapter(data,view){rendered.push(data.label);attachReviewActions(view);}
function setNotice(){}
`,ctx);
vm.runInContext(cut('function clearChapterView(', 'function openRun('),ctx);
if(src.includes('function isCurrentChapter('))
 vm.runInContext(cut('function isCurrentChapter(', 'async function loadChapter()'),ctx);
vm.runInContext(cut('async function loadChapter()', 'function renderChapter('),ctx);
vm.runInContext(cut('function attachReviewActions(', 'el("btn-load-chapter")'),ctx);
(async()=>{
 vm.runInContext("attachReviewActions({runId:'A',chapter:1,generation:0,chapterGeneration:0});"+
  "el('review-desk').innerHTML='old chapter-1 controls';el('chapter-number').value='2'",ctx);
 const oldCallback=vm.runInContext('callback',ctx);
 const second=vm.runInContext('loadChapter()',ctx);
 assert.equal(vm.runInContext("el('review-desk').innerHTML",ctx),'');
 assert.equal(vm.runInContext("el('chapter-number').value",ctx),'2');
 const staleAction=oldCallback();
 assert.equal(vm.runInContext('writes.length',ctx),0);await staleAction;
 vm.runInContext("pending[0].resolve({run_id:'A',chapter:2,label:'two'})",ctx);await second;
 // A rendered current control still sends the correct CAS edit.
 const action=vm.runInContext('callback()',ctx);await Promise.resolve();
 assert.equal(vm.runInContext('writes[0].path',ctx),'/runs/A/chapters/2/chunks/shared-chunk/final');
 assert.equal(vm.runInContext('writes[0].body.expected_machine_sha256',ctx),'shared-machine');
 await Promise.resolve();
 vm.runInContext("pending[1].resolve({run_id:'A',chapter:2,label:'after-edit'})",ctx);await action;
 // A late response for the same chapter must not replace a newer response.
 vm.runInContext('rendered.length=0',ctx);
 const older=vm.runInContext('loadChapter()',ctx);
 const newer=vm.runInContext('loadChapter()',ctx);
 vm.runInContext("pending[3].resolve({run_id:'A',chapter:2,label:'new'})",ctx);await newer;
 vm.runInContext("pending[2].resolve({run_id:'A',chapter:2,label:'old'})",ctx);await older;
 assert.equal(vm.runInContext('JSON.stringify(rendered)',ctx),'["new"]');
 // Returning to chapter 1 does not authorize the first chapter-1 request.
 vm.runInContext("rendered.length=0;el('chapter-number').value='1'",ctx);
 const first1=vm.runInContext('loadChapter()',ctx);
 vm.runInContext("el('chapter-number').value='2'",ctx);
 const middle2=vm.runInContext('loadChapter()',ctx);
 vm.runInContext("el('chapter-number').value='1'",ctx);
 const last1=vm.runInContext('loadChapter()',ctx);
 vm.runInContext("pending[6].resolve({run_id:'A',chapter:1,label:'last1'})",ctx);await last1;
 vm.runInContext("pending[4].resolve({run_id:'A',chapter:1,label:'first1'})",ctx);await first1;
 vm.runInContext("pending[5].resolve({run_id:'A',chapter:2,label:'middle2'})",ctx);await middle2;
 assert.equal(vm.runInContext('JSON.stringify(rendered)',ctx),'["last1"]');
 console.log(JSON.stringify({ok:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run(
        ["node", "-e", script, str(ROOT / "src/novel_harness/web/app.js")],
        check=False, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}
