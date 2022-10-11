const jsonServer = require('json-server');
const server = jsonServer.create();
const router = jsonServer.router('db.json');
const middlewares = jsonServer.defaults();

server.use(middlewares);
server.use(router);

// Labo PC
server.listen(3001, '192.168.2.111', () => {
  console.log('run');
});


// // note PC - 3H
// server.listen(3001, '192.168.2.19', () => {
//   console.log('run');
// });


// // note PC - 4K
// server.listen(3001, '172.19.0.158', () => {
//   console.log('run');
// });