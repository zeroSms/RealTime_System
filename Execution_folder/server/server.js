const jsonServer = require('json-server');
const server = jsonServer.create();
const router = jsonServer.router('db.json');
const middlewares = jsonServer.defaults();

server.use(middlewares);
server.use(router);

// // Labo PC
// server.listen(3000, '192.168.2.111', () => {
//   console.log('run');
// });


// note PC
server.listen(3000, '192.168.2.19', () => {
  console.log('run');
});