// Execute the real page script with synthetic WebXR and transport adapters.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'quest_controller.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = {};
const requests = [];
const transforms = (position, orientation={x:0,y:0,z:0,w:1}) => ({position, orientation});
const gl = new Proxy({}, {get: (_, name) => {
  if (name === 'getShaderParameter' || name === 'getProgramParameter') return () => true;
  if (name.startsWith('create')) return () => ({});
  return () => {};
}});
const ctx = vm.createContext({
  document: {getElementById: id => elements[id] ||= {addEventListener() {}, getContext: () => gl}},
  window: {addEventListener() {}},
  navigator: {}, console, setInterval() {},
  performance: {now: () => 1000},
  AbortSignal: {timeout: () => ({})},
  fetch: (url, options={}) => {
    requests.push({url, options});
    return Promise.resolve({ok:true, blob:async () => ({})});
  },
  createImageBitmap: async () => ({close() {}}),
});
vm.runInContext(script, ctx);
ctx.fakeGL = gl;
vm.runInContext(`
  gl = fakeGL;
  videoReceivedAt = 1000;
  session = {
    visibilityState:'visible', requestAnimationFrame() {},
    renderState:{baseLayer:{framebuffer:{}, getViewport:()=>({x:0,y:0,width:100,height:100})}},
    inputSources: [{
      handedness:'left', gripSpace:'left',
      gamepad:{mapping:'xr-standard',axes:[0,0,.2,-.8],buttons:[{}, {value:.9}, {}, {pressed:true}]}
    }, {
      handedness:'right', gripSpace:'right',
      gamepad:{mapping:'xr-standard',axes:[0,0,-.4,0],buttons:[{}, {value:0}, {}, {pressed:false}]}
    }]
  };
`, ctx);
ctx.frame = {
  getViewerPose: () => ({transform:transforms({x:0,y:1.6,z:0}), views:[{},{}]}),
  getPose: space => ({transform:transforms({x:space === 'left' ? -.2 : .2,y:1.2,z:-.3})}),
};
const flush = () => new Promise(resolve => setImmediate(resolve));
const packets = () => requests.filter(r => r.url === '/input').map(r => JSON.parse(r.options.body));
(async () => {
  vm.runInContext('onFrame(1000, frame)', ctx);
  await flush();
  let p = packets().at(-1);
  assert.equal(p.active, true);
  assert.deepEqual(p.hands.left.stick, [.2,-.8]);
  assert.deepEqual(p.hands.right.stick, [-.4,0]);
  assert.equal(p.hands.left.squeeze, .9);
  assert.equal(p.recenter, true);
  assert.deepEqual(p.head.position, [0,1.6,0]);
  vm.runInContext('onFrame(1040, frame)', ctx);
  await flush();
  assert.equal(packets().at(-1).recenter, false, 'held thumbstick should not repeatedly recenter');
  vm.runInContext("session.visibilityState='hidden'; onFrame(1080, frame)", ctx);
  await flush();
  assert.equal(packets().at(-1).active, false, 'hidden VR must pause');
  vm.runInContext("session.visibilityState='visible'; videoReceivedAt=-Infinity; onFrame(1120, frame)", ctx);
  await flush();
  assert.equal(packets().at(-1).active, false, 'stale camera must pause');
  vm.runInContext('pending=true; send({},false)', ctx);
  await flush();
  assert.equal(packets().at(-1).active, false, 'stop must bypass an in-flight motion request');
  console.log('WebXR mapping, recenter, visibility, stale-video and stop checks passed.');
})().catch(error => { console.error(error); process.exitCode=1; });
