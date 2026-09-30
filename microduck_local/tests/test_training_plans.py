"""Plan boundaries: import, immutable revisions, independent launch stacks, AI drafts."""
import asyncio
import json
import types

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from microduck_local import behaviors as B
from microduck_local import viz_server as V
from microduck_local import training_plans as P


def endpoint(app, path, method):
    return next(r.endpoint for r in app.routes if getattr(r, 'path', '') == path and method in (getattr(r, 'methods', None) or []))


def request(origin=None):
    return Request({'type':'http', 'headers': [] if origin is None else [(b'origin', origin.encode())]})


def plan():
    return {'version':1, 'title':'单脚平衡', 'description':'学会稳定单脚站立', 'success':'多种初始姿态下稳定站立',
            'robot':'microduck', 'source':'manual', 'local':{'behavior':'one_leg','steps':100000,'weights':{'smooth_moves':4}},'cloud':None}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv('MICRODUCK_PLANS_DIR', str(tmp_path/'plans'))
    monkeypatch.setenv('MICRODUCK_CLIPS_DIR', str(tmp_path/'clips'))
    a=FastAPI()
    calls=[]
    async def local(req):
        calls.append(('local',req))
        return {'matched':True,'job':{'id':'fake-local'}}
    def cloud(req, request):
        calls.append(('cloud',req))
        return {'id':'fake-cloud'}
    P.mount_plans(a, behaviors=B, clean_clip=V.clean_clip, atomic_write=V._atomic_write_json,
                  launch_local=local, launch_colab=cloud, launch_hf=cloud,
                  teach_req=V.TeachReq, colab_req=V.ColabStartReq, hf_req=V.HfStartReq,
                  clip_path=V.clip_path, origin_allowed=V.origin_allowed)
    a.calls=calls
    return a


def create(a,p=None):
    return endpoint(a,'/plans','POST')(p or plan(),request())


def test_version_conflict_and_old_version(app):
    first=create(app)
    data={k:v for k,v in first.items() if k not in ('path','storage','revisions')}
    data['title']='修改后的版本'
    second=endpoint(app,'/plans/{pid}','PUT')(first['id'],data,request())
    assert second['revision']==2
    old=endpoint(app,'/plans/{pid}/versions/{revision}','GET')(first['id'],1)
    assert old['title']=='单脚平衡'
    assert old['revisions']==[1,2]
    assert old['path'].endswith('/v1.json')
    with pytest.raises(HTTPException) as exc:
        endpoint(app,'/plans/{pid}','PUT')(first['id'],data,request())
    assert exc.value.status_code==409


def test_import_python_is_literal_only(app,tmp_path):
    code=f'import os\nos.remove({str(tmp_path / "important")!r})\nTRAINING_PLAN = {plan()!r}'
    marker=tmp_path/'important';marker.write_text('keep')
    result=endpoint(app,'/plans/import','POST')(P.ImportReq(filename='training.py',content=code),request())
    assert marker.exists()
    assert result['source']=='import'
    with pytest.raises(ValueError):
        P.parse_file('bad.py','TRAINING_PLAN = __import__("os").system("false")')


@pytest.mark.parametrize('filename,content',[('weights.onnx','x'),('plan.json','[]'),('plan.json','{}'),('plan.json','not-json'),('x.py','print("hello")')])
def test_wrong_file_reports_error(app,filename,content):
    with pytest.raises(HTTPException) as exc:
        endpoint(app,'/plans/import','POST')(P.ImportReq(filename=filename,content=content),request())
    assert exc.value.status_code==422


@pytest.mark.parametrize('weights',[{'unknown':2},{'smooth_moves':-1},{'smooth_moves':float('nan')},{'smooth_moves':True}])
def test_bad_reward_weights(app,weights):
    p=plan();p['local']['weights']=weights
    with pytest.raises(HTTPException) as exc:create(app,p)
    assert exc.value.status_code==422


def test_local_launch_pins_all_weights_and_does_not_inherit_sticky(app):
    saved=create(app)
    result=asyncio.run(endpoint(app,'/plans/{pid}/launch','POST')(saved['id'],P.LaunchReq(revision=1,compute='local'),request()))
    assert result['started']
    kind,req=app.calls[0]
    assert kind=='local'
    assert req.weights['smooth_moves']==4
    assert req.weights['one_leg_hold']==3
    assert req.stageSteps=={} and req.stageWeights=={}
    assert req.steps==100000
    receipts=list(P.root().glob('*/launch-*.json'))
    assert json.loads(receipts[0].read_text())['plan']['revision']==1


def test_cloud_requires_independent_config(app):
    p=create(app)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(endpoint(app,'/plans/{pid}/launch','POST')(p['id'],P.LaunchReq(revision=1,compute='hf'),request()))
    assert exc.value.status_code==422 and not app.calls
    data=plan();data['cloud']={'task':'Mjlab-Velocity-Flat-MicroDuck','iterations':7,'envs':64,'gpu':'T4','flavor':'l4x1'}
    saved=create(app,data)
    asyncio.run(endpoint(app,'/plans/{pid}/launch','POST')(saved['id'],P.LaunchReq(revision=1,compute='hf'),request()))
    _,req=app.calls[0]
    assert req.task=='Mjlab-Velocity-Flat-MicroDuck' and req.iterations==7
    assert not hasattr(req,'weights')


def clip():
    return {'version':1,'name':'my-motion','robot':'microduck','duration':1,'loop':False,
            'keys':[{'t':0,'joints':[0]*14,'rootPitch':0},{'t':1,'joints':[0]*14,'rootPitch':0}]}


def test_clip_is_embedded_and_launched_as_version_snapshot(app):
    p=plan();p['local']['behavior']='imitate';p['local']['weights']={};p['reference']=clip()
    saved=create(app,p)
    p['reference']['keys'][0]['joints'][0]=1
    asyncio.run(endpoint(app,'/plans/{pid}/launch','POST')(saved['id'],P.LaunchReq(revision=1,compute='local'),request()))
    name=app.calls[0][1].clip
    snapshot=json.loads(V.clip_path(name).read_text())
    assert snapshot['keys'][0]['joints'][0]==0
    assert name==f"plan-{saved['id']}-v1"


def test_clip_cloud_and_robot_mismatch_rejected(app):
    p=plan();p['local']['behavior']='imitate';p['local']['weights']={};p['reference']=clip()
    p['cloud']={'task':'Mjlab-Velocity-Flat-MicroDuck','iterations':1,'envs':64}
    with pytest.raises(HTTPException):create(app,p)
    p['cloud']=None;p['reference']['robot']='g1'
    with pytest.raises(HTTPException):create(app,p)


def test_path_traversal_and_foreign_origin(app):
    with pytest.raises(HTTPException):endpoint(app,'/plans/{pid}/versions/{revision}','GET')('../outside',1)
    with pytest.raises(HTTPException) as exc:endpoint(app,'/plans','POST')(plan(),request('https://untrusted.example'))
    assert exc.value.status_code==403


def test_ai_key_not_returned_and_draft_not_saved(app,monkeypatch):
    endpoint(app,'/plans/ai/settings','PUT')(P.AiSettings(base_url='http://localhost:11434/v1',model='local',api_key='private'),request())
    status=endpoint(app,'/plans/ai/settings','GET')()
    assert 'api_key' not in status and status['configured']
    assert (P.root()/'ai-settings.json').stat().st_mode & 0o777 == 0o600
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'choices':[{'message':{'content':json.dumps(plan())}}]}).encode()
    monkeypatch.setattr(P,'urlopen',lambda *args,**kw:Response())
    draft=endpoint(app,'/plans/ai/draft','POST')(P.AiReq(prompt='学习单脚站立'),request())
    assert draft['draft']['source']=='ai'
    assert not list(P.root().glob('*/plan.json')) and not app.calls


def test_budget_cannot_be_silently_clamped(app):
    p=plan();p['local']['steps']=8192
    with pytest.raises(HTTPException):create(app,p)


def test_cloud_preflight_reports_missing_teachers(app,monkeypatch):
    from microduck_local import colab_jobs
    p=plan();p['cloud']={'task':'Mjlab-VelStand-Flat-MicroDuck','iterations':1,'envs':64}
    saved=create(app,p)
    def missing(task):raise RuntimeError('缺少教师模型')
    monkeypatch.setattr(colab_jobs,'teacher_checkpoints',missing)
    result=endpoint(app,'/plans/{pid}/check','POST')(saved['id'],P.LaunchReq(revision=1,compute='hf'),request())
    assert not result['ready'] and result['problems']==['缺少教师模型'] and not app.calls


def test_latest_plan_tracks_saved_revision_and_preserves_pinned_version(app):
    first = create(app)
    changed = {k: v for k, v in first.items() if k not in ('path', 'storage', 'revisions')}
    changed['title'] = '最新版本'
    second = endpoint(app, '/plans/{pid}', 'PUT')(first['id'], changed, request())
    latest = endpoint(app, '/plans/{pid}', 'GET')(first['id'])
    assert latest['revision'] == second['revision'] == 2
    assert latest['title'] == '最新版本'
    assert latest['path'].endswith('/v2.json')
    old = endpoint(app, '/plans/{pid}/versions/{revision}', 'GET')(first['id'], 1)
    assert old['title'] == first['title']
    assert old['revision'] == 1


@pytest.mark.parametrize('pid,status', [('missing', 404), ('../escape', 422)])
def test_latest_plan_validates_identifier_and_missing_plan(app, pid, status):
    with pytest.raises(HTTPException) as exc:
        endpoint(app, '/plans/{pid}', 'GET')(pid)
    assert exc.value.status_code == status


def test_latest_plan_route_does_not_shadow_templates(app):
    from starlette.routing import Match
    scope = {'type': 'http', 'path': '/plans/templates', 'method': 'GET'}
    matched = next(route for route in app.routes if route.matches(scope)[0] == Match.FULL)
    assert matched.path == '/plans/templates'
