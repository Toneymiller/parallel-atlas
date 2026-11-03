'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const host = '127.0.0.1';
const port = Number(process.env.PORT || 4173);
const publicFiles = new Set(['index.html', 'styles.css', 'app.js', 'engine.js', 'models.js', 'hardware.js', '.nojekyll']);
const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  console.error('PORT must be an integer between 1 and 65535.');
  process.exit(1);
}

function respond(res, status, message) {
  res.writeHead(status, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(message);
}

async function handleRequest(req, res) {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.setHeader('Allow', 'GET, HEAD');
    return respond(res, 405, 'Method not allowed');
  }

  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, `http://${host}:${port}`).pathname);
  } catch {
    return respond(res, 400, 'Invalid request path');
  }

  if (pathname.includes('\0') || pathname.includes('\\')) {
    return respond(res, 400, 'Invalid request path');
  }

  const name = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
  const parts = name.split('/');
  const allowed = publicFiles.has(name) || (parts[0] === 'assets' && parts.length > 1);
  if (!allowed || parts.some(part => part === '..' || (part.startsWith('.') && name !== '.nojekyll'))) {
    return respond(res, 404, 'Not found');
  }

  const filename = path.resolve(root, name);
  const relative = path.relative(root, filename);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    return respond(res, 404, 'Not found');
  }

  try {
    const realFilename = await fs.promises.realpath(filename);
    const realRelative = path.relative(root, realFilename);
    if (realRelative.startsWith('..') || path.isAbsolute(realRelative)) {
      return respond(res, 404, 'Not found');
    }
    const stat = await fs.promises.stat(realFilename);
    if (!stat.isFile()) return respond(res, 404, 'Not found');

    res.writeHead(200, {
      'Content-Type': contentTypes[path.extname(realFilename).toLowerCase()] || 'application/octet-stream',
      'Content-Length': stat.size,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    if (req.method === 'HEAD') return res.end();
    const stream = fs.createReadStream(realFilename);
    stream.on('error', () => res.destroy());
    stream.pipe(res);
  } catch (error) {
    respond(res, error.code === 'ENOENT' || error.code === 'ENOTDIR' ? 404 : 500,
      error.code === 'ENOENT' || error.code === 'ENOTDIR' ? 'Not found' : 'Unable to read file');
  }
}

const server = http.createServer((req, res) => {
  handleRequest(req, res).catch(() => {
    if (!res.headersSent) respond(res, 500, 'Unable to serve request');
    else res.destroy();
  });
});

server.on('error', error => {
  console.error(error.code === 'EADDRINUSE'
    ? `Port ${port} is already in use. Set PORT to choose another port.`
    : `Unable to start local server: ${error.message}`);
  process.exitCode = 1;
});

server.listen(port, host, () => {
  console.log(`Parallel Atlas: http://${host}:${port}`);
  console.log('Press Ctrl+C to stop.');
});
