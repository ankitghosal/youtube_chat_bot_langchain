const state = { videoId: null };
const $ = (selector) => document.querySelector(selector);

function setStatus(text, active = false) {
    $('#statusText').textContent = text;
    $('.status-lamp span:first-child').style.background = active ? 'var(--orange)' : 'var(--acid)';
}

function addMessage(role, content, timestamps = []) {
    $('#emptyState').hidden = true;
    const message = document.createElement('article');
    message.className = `message ${role}`;
    const tag = document.createElement('span');
    tag.className = 'message-tag';
    tag.textContent = role === 'user' ? 'YOU' : 'AGENT';
    message.append(tag, document.createTextNode(content));
    if (timestamps.length) {
        const links = document.createElement('div');
        links.className = 'timestamps';
        timestamps.forEach(({ label, url }) => {
            const link = document.createElement('a');
            link.href = url; link.target = '_blank'; link.rel = 'noreferrer'; link.textContent = `▶ ${label}`;
            links.append(link);
        });
        message.append(links);
    }
    $('#messages').append(message);
    message.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function requestJson(url, options) {
    const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'The terminal returned an error.');
    return body;
}

$('#loadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = $('#loadButton'); button.disabled = true; setStatus('INDEXING', true);
    state.videoId = null;
    $('#sendButton').disabled = true;
    $('#videoMeta').hidden = true;
    try {
        const data = await requestJson('/api/load', { method: 'POST', body: JSON.stringify({ video: $('#videoInput').value }) });
        state.videoId = data.video_id;
        $('#videoMeta').innerHTML = `<strong>VIDEO ${data.video_id}</strong> // ${data.chunks} chunks indexed<br>${data.preview}...`;
        $('#videoMeta').hidden = false; $('#chatForm').hidden = false; $('#clearButton').hidden = false;
        $('#sendButton').disabled = false;
        $('#messages').replaceChildren(); $('#emptyState').hidden = false; $('#emptyState').textContent = 'Archive ready. Ask your first question.';
        setStatus('ONLINE');
    } catch (error) { setStatus('ERROR', true); $('#videoMeta').textContent = error.message; $('#videoMeta').hidden = false; }
    finally { button.disabled = false; }
});

$('#chatForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state.videoId) return;
    const input = $('#questionInput'); const question = input.value.trim(); if (!question) return;
    addMessage('user', question); input.value = ''; $('#sendButton').disabled = true; setStatus('THINKING', true);
    try {
        const data = await requestJson('/api/chat', { method: 'POST', body: JSON.stringify({ video_id: state.videoId, question }) });
        addMessage('assistant', data.answer, data.timestamps); setStatus('ONLINE');
    } catch (error) { addMessage('assistant', `ERROR: ${error.message}`); setStatus('ERROR', true); }
    finally { $('#sendButton').disabled = false; input.focus(); }
});

$('#clearButton').addEventListener('click', () => { $('#messages').replaceChildren(); $('#emptyState').hidden = false; $('#emptyState').textContent = 'Archive ready. Ask your first question.'; });
$('#clock').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
