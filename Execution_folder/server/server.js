const jsonServer = require('json-server');
const server = jsonServer.create();
const router = jsonServer.router('db.json');
const middlewares = jsonServer.defaults();

server.use(middlewares);
server.use(router);


server_address = '192.168.2.120'
// server_address = '192.168.2.22'   // dlcl_1, notePC
// server_address = '172.19.0.44'    // cs-wlan-g, notePC
// server_address = '192.168.0.2'  // 4H - aterm, notePC


server.listen(3001, server_address, () => {
  console.log('run');
});
