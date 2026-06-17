import http from 'http';

const PORT = process.env.PORT || 7860;

// Anti-crash global obligatoire
process.on('uncaughtException', (err) => console.error('CRASH ÉVITÉ:', err));
process.on('unhandledRejection', (err) => console.error('PROMESSE REFUSÉE:', err));

const LANDING_PAGE = `
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Smart Proxy</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #0f172a; color: #f8fafc; margin: 0; }
        .container { text-align: center; background: #1e293b; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); width: 90%; max-width: 500px; }
        h1 { color: #38bdf8; margin-bottom: 10px; font-size: 28px; }
        p { color: #94a3b8; margin-bottom: 30px; font-size: 14px; }
        form { display: flex; flex-direction: column; gap: 16px; }
        input { padding: 14px; border: 2px solid #334155; border-radius: 8px; background: #0f172a; color: white; font-size: 16px; outline: none; transition: border-color 0.2s; }
        input:focus { border-color: #38bdf8; }
        button { padding: 14px; border: none; border-radius: 8px; background: #38bdf8; color: #0f172a; font-size: 16px; cursor: pointer; font-weight: bold; transition: background 0.2s; }
        button:hover { background: #7dd3fc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Proxy Universel</h1>
        <p>Entre n'importe quelle URL (Ex: http://gamingserver.fr:25578 ou https://google.com)</p>
        <form action="/" method="GET">
            <input type="url" name="url" placeholder="https://example.com" required>
            <button type="submit">Visiter le site</button>
        </form>
    </div>
</body>
</html>
`;

function parseCookies(cookieHeader) {
    const list = {};
    if (!cookieHeader) return list;
    cookieHeader.split(';').forEach(cookie => {
        const parts = cookie.split('=');
        list[parts.shift().trim()] = decodeURIComponent(parts.join('='));
    });
    return list;
}

function resolveTarget(req, reqUrl) {
    let target = reqUrl.searchParams.get('url');
    if (target) return { source: 'query', url: target };

    if (req.headers.referer) {
        try {
            const refUrl = new URL(req.headers.referer);
            let refTarget = refUrl.searchParams.get('url');
            if (refTarget) {
                const base = new URL(refTarget);
                return { source: 'referer', url: new URL(req.url, base.origin).href };
            }
        } catch (e) {}
    }

    const cookies = parseCookies(req.headers.cookie);
    if (cookies.proxy_last_origin) {
        try {
            const baseOrigin = decodeURIComponent(cookies.proxy_last_origin);
            return { source: 'cookie', url: new URL(req.url, baseOrigin).href };
        } catch (e) {}
    }

    return null;
}

const server = http.createServer(async (req, res) => {
    const host = req.headers.host || '127.0.0.1';
    
    if (req.url === '/favicon.ico') {
        res.writeHead(204);
        return res.end();
    }

    const reqUrl = new URL(req.url, `http://${host}`);
    const targetData = resolveTarget(req, reqUrl);

    if (!targetData) {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        return res.end(LANDING_PAGE);
    }

    let targetUrlStr = targetData.url;

    try {
        if (!targetUrlStr.startsWith('http://') && !targetUrlStr.startsWith('https://')) {
            targetUrlStr = 'https://' + targetUrlStr;
        }
        const targetUrl = new URL(targetUrlStr);

        const headers = new Headers();
        for (const [key, value] of Object.entries(req.headers)) {
            if (!['host', 'cookie', 'accept-encoding', 'connection', 'origin', 'referer'].includes(key.toLowerCase())) {
                headers.set(key, value);
            }
        }
        
        headers.set('Host', targetUrl.host);
        headers.set('User-Agent', req.headers['user-agent'] || 'Mozilla/5.0');
        if (req.headers['origin']) headers.set('Origin', targetUrl.origin);
        if (req.headers['referer']) headers.set('Referer', targetUrl.origin);

        if (req.headers.cookie) {
            const cleanCookies = req.headers.cookie.split(';')
                .map(c => c.trim())
                .filter(c => !c.startsWith('proxy_last_origin='))
                .join('; ');
            if (cleanCookies) headers.set('Cookie', cleanCookies);
        }

        let body = null;
        if (req.method !== 'GET' && req.method !== 'HEAD') {
            const buffers = [];
            for await (const chunk of req) {
                buffers.push(chunk);
            }
            body = Buffer.concat(buffers);
        }

        const response = await fetch(targetUrl.href, {
            method: req.method,
            headers: headers,
            body: body,
            redirect: 'manual'
        });

        const contentType = response.headers.get('content-type') || '';
        const isHtml = contentType.includes('text/html');

        // Base des en-têtes système à nettoyer
        const stripHeaders = [
            'connection', 'keep-alive', 'transfer-encoding', 
            'content-encoding', 'set-cookie', 'location', 
            'content-security-policy', 'strict-transport-security'
        ];

        const resHeaders = {};
        response.headers.forEach((value, key) => {
            const lowerKey = key.toLowerCase();
            if (!stripHeaders.includes(lowerKey)) {
                // CORRECTION FIX TÉLÉCHARGEMENT : On ne supprime la taille QUE si c'est du HTML modifié
                if (lowerKey === 'content-length' && isHtml) return;
                resHeaders[key] = value;
            }
        });

        const setCookies = [`proxy_last_origin=${encodeURIComponent(targetUrl.origin)}; Path=/; HttpOnly; SameSite=Lax`];
        const originalSetCookies = response.headers.getSetCookie ? response.headers.getSetCookie() : [];
        originalSetCookies.forEach(c => setCookies.push(c));
        resHeaders['Set-Cookie'] = setCookies;

        if (response.status >= 300 && response.status < 400) {
            let location = response.headers.get('location');
            if (location) {
                const absoluteLocation = new URL(location, targetUrl.href).href;
                resHeaders['Location'] = `/?url=${encodeURIComponent(absoluteLocation)}`;
            }
            res.writeHead(response.status, resHeaders);
            return res.end();
        }

        if (isHtml) {
            let text = await response.text();
            text = text.replace(/(href|src)=["'](https?:\/\/[^"']+)["']/g, (match, attr, url) => {
                return `${attr}="/?url=${encodeURIComponent(url)}"`;
            });

            const bodyBuffer = Buffer.from(text, 'utf-8');
            resHeaders['content-length'] = bodyBuffer.length;
            res.writeHead(response.status, resHeaders);
            return res.end(bodyBuffer);
        } 
        
        // CORRECTION FLUX : Boucle d'écriture directe paquet par paquet pour éviter les pertes sur les fichiers/téléchargements
        res.writeHead(response.status, resHeaders);
        if (response.body) {
            try {
                for await (const chunk of response.body) {
                    res.write(chunk);
                }
            } catch (streamErr) {
                console.error("Erreur durant le transfert du fichier:", streamErr.message);
            }
        }
        res.end();

    } catch (err) {
        console.error(`[Erreur Cible] ${targetUrlStr}:`, err.message);
        if (!res.headersSent) {
            res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
            res.end(`Erreur Proxy : Impossible de joindre la cible.`);
        }
    }
});

server.on('upgrade', (req, socket, head) => {
    const host = req.headers.host || '127.0.0.1';
    const reqUrl = new URL(req.url, `http://${host}`);
    const targetData = resolveTarget(req, reqUrl);

    if (!targetData) {
        socket.end();
        return;
    }

    const target = new URL(targetData.url);
    console.log(`[WS Universel] Connexion injectée vers : ${target.host}`);

    const options = {
        hostname: target.hostname,
        port: target.port || (target.protocol === 'https:' ? 443 : 80),
        method: req.method,
        path: req.url,
        headers: { ...req.headers }
    };

    options.headers['host'] = target.host;
    if (options.headers['origin']) options.headers['origin'] = target.origin;

    const proxyReq = http.request(options);
    proxyReq.on('upgrade', (proxyRes, proxySocket, proxyHead) => {
        socket.write('HTTP/1.1 101 Switching Protocols\r\n');
        for (const [key, value] of Object.entries(proxyRes.headers)) {
            socket.write(`${key}: ${value}\r\n`);
        }
        socket.write('\r\n');

        proxySocket.write(proxyHead);
        proxySocket.pipe(socket);
        socket.pipe(proxySocket);
    });

    proxyReq.on('error', () => socket.end());
    proxyReq.end();
});

server.listen(PORT, '0.0.0.0', () => {
    console.log(`Proxy Multi-Site Intelligent & Téléchargements corrigés sur le port ${PORT}`);
});