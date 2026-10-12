/**
 * AI DR Screen — Context-Aware Chatbot Assistant Frontend Controller
 * Integrates smoothly with Master Portal, 5-Stage Pipeline, and Screening Results.
 */

(function() {
    'use strict';

    // Prevent duplicate mounting if loaded in iframes
    if (window.self !== window.top) {
        return;
    }

    const API_ENDPOINT = '/api/chat';
    let sessionId = localStorage.getItem('aidr_chat_session_id');
    if (!sessionId) {
        sessionId = 'sess_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
        localStorage.setItem('aidr_chat_session_id', sessionId);
    }

    let currentLanguage = localStorage.getItem('aidr_chat_lang') || 'en';
    let isWaitingResponse = false;

    // Elements
    let triggerBtn = null;
    let chatPanel = null;
    let messagesContainer = null;
    let inputField = null;
    let sendBtn = null;
    let langSelect = null;

    document.addEventListener('DOMContentLoaded', () => {
        initChatbotUI();
        initContextListeners();
    });

    function initChatbotUI() {
        if (document.getElementById('aidr-chat-trigger')) return;

        // 1. Create Floating Trigger
        triggerBtn = document.createElement('button');
        triggerBtn.id = 'aidr-chat-trigger';
        triggerBtn.setAttribute('aria-label', 'Open Oculyn AI Assistant');
        triggerBtn.title = 'Oculyn — AI Assistant for DR Screening';
        triggerBtn.innerHTML = `
            <div class="chat-pulse-ring"></div>
            <div class="chat-trigger-avatar-wrap">
                <img src="oculyn_avatar.png" alt="Oculyn AI Assistant" class="chat-trigger-avatar-img">
                <span class="chat-trigger-status-dot"></span>
            </div>
            <div class="chat-trigger-close-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
            </div>
        `;

        // 2. Create Chat Panel
        chatPanel = document.createElement('div');
        chatPanel.id = 'aidr-chat-panel';
        chatPanel.setAttribute('role', 'dialog');
        chatPanel.setAttribute('aria-label', 'Oculyn AI DR Screen Assistant');
        chatPanel.innerHTML = `
            <div class="aidr-chat-header">
                <div class="aidr-chat-title-wrap">
                    <div class="aidr-chat-avatar">
                        <img src="oculyn_avatar.png" alt="Oculyn" class="aidr-header-avatar-img">
                        <span class="aidr-header-status-dot"></span>
                    </div>
                    <div>
                        <div class="aidr-chat-title-row">
                            <h3 class="aidr-chat-title">Oculyn</h3>
                            <span class="aidr-badge-ai">✨ AI Assistant</span>
                        </div>
                        <div class="aidr-chat-subtitle">
                            <span class="online-dot"></span>
                            <span>Diabetic Retinopathy Screening Guide</span>
                        </div>
                    </div>
                </div>
                <div class="aidr-chat-header-actions">
                    <select class="aidr-chat-lang-select" id="aidr-chat-lang" title="Select Language">
                        <option value="en" ${currentLanguage === 'en' ? 'selected' : ''}>English</option>
                        <option value="ta" ${currentLanguage === 'ta' ? 'selected' : ''}>தமிழ்</option>
                        <option value="hi" ${currentLanguage === 'hi' ? 'selected' : ''}>हिंदी</option>
                    </select>
                    <button class="aidr-chat-btn-icon" id="aidr-chat-btn-clear" title="Clear Conversation">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
                    </button>
                    <button class="aidr-chat-btn-icon" id="aidr-chat-btn-close" title="Close Assistant">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
                    </button>
                </div>
            </div>

            <div class="aidr-chat-quick-actions" id="aidr-quick-actions">
                <button class="aidr-quick-chip" data-action="start_screening">👁 Retinal Screening</button>
                <button class="aidr-quick-chip" data-action="symptom_checker">🩺 Symptom Checker</button>
                <button class="aidr-quick-chip" data-action="upload_image">📷 Upload Eye Image</button>
                <button class="aidr-quick-chip" data-action="explain_ai">🔍 Explain AI Result</button>
                <button class="aidr-quick-chip" data-action="understand_result">📊 DR Grades (0–4)</button>
                <button class="aidr-quick-chip" data-action="find_healthcare">🏥 Find a Doctor</button>
                <button class="aidr-quick-chip" data-action="upload_report">📄 Upload Report</button>
                <button class="aidr-quick-chip" data-action="health_insights">💡 Health Insights</button>
                <button class="aidr-quick-chip" data-action="navigate">🧭 Navigate Portal</button>
                <button class="aidr-quick-chip" data-action="support">🛡️ 24/7 Support</button>
            </div>

            <div class="aidr-chat-messages" id="aidr-chat-messages">
                <!-- Welcome Hero Card with Oculyn illustration -->
                <div class="aidr-welcome-card">
                    <div class="aidr-welcome-banner-wrap">
                        <img src="oculyn_banner.jpg" alt="Oculyn AI Assistant" class="aidr-welcome-banner-img">
                        <div class="aidr-welcome-banner-badge">✨ Meet Oculyn</div>
                    </div>
                    <div class="aidr-welcome-card-body">
                        <h4>AI Assistant for DR Screening</h4>
                        <p>Tell me how you feel, I'm here to help! Ask me anything about screening, upload guidance, Grad-CAM heatmaps, or finding local eye hospitals.</p>
                    </div>
                </div>

                <!-- Welcome greeting -->
                <div class="aidr-msg assistant">
                    <img src="oculyn_avatar.png" alt="Oculyn" class="aidr-msg-avatar">
                    <div class="aidr-msg-content">
                        <div class="aidr-msg-bubble">
                            Hi! I'm <strong>Oculyn</strong> 👋, your AI Assistant for <strong>AI DR Screen</strong>.<br><br>
                            Tell me how you feel, I'm here to help! I can guide you through the 5 screening stations, upload retinal images or lab reports, explain DR grades (0–4), Grad-CAM heatmaps, and find nearby eye hospitals.<br><br>
                            <em>How can I assist you today?</em>
                        </div>
                        <div class="aidr-msg-meta">
                            <span>Oculyn • AI Assistant</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="aidr-chat-input-area">
                <input type="text" class="aidr-chat-input" id="aidr-chat-input" placeholder="Ask Oculyn anything..." autocomplete="off">
                <button class="aidr-chat-btn-send" id="aidr-chat-btn-send" title="Send Message">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
            </div>
        `;

        document.body.appendChild(triggerBtn);
        document.body.appendChild(chatPanel);

        // Bind DOM elements
        messagesContainer = document.getElementById('aidr-chat-messages');
        inputField = document.getElementById('aidr-chat-input');
        sendBtn = document.getElementById('aidr-chat-btn-send');
        langSelect = document.getElementById('aidr-chat-lang');

        // Event listeners
        triggerBtn.addEventListener('click', toggleChatPanel);
        document.getElementById('aidr-chat-btn-close').addEventListener('click', closeChatPanel);
        document.getElementById('aidr-chat-btn-clear').addEventListener('click', clearChatSession);

        sendBtn.addEventListener('click', handleSendMessage);
        inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleSendMessage();
            }
        });

        langSelect.addEventListener('change', (e) => {
            currentLanguage = e.target.value;
            localStorage.setItem('aidr_chat_lang', currentLanguage);
            appendAssistantMessage(getLanguageSwitchGreeting(currentLanguage));
        });

        // Quick action chip clicks
        const chips = document.querySelectorAll('.aidr-quick-chip');
        chips.forEach(chip => {
            chip.addEventListener('click', () => {
                const action = chip.getAttribute('data-action');
                handleQuickAction(action, chip.textContent.trim());
            });
        });
    }

    function toggleChatPanel() {
        if (!chatPanel) return;
        const isOpen = chatPanel.classList.toggle('open');
        if (triggerBtn) {
            triggerBtn.classList.toggle('panel-open', isOpen);
        }
        if (isOpen && inputField) {
            setTimeout(() => inputField.focus(), 150);
        }
    }

    function closeChatPanel() {
        if (chatPanel) {
            chatPanel.classList.remove('open');
        }
        if (triggerBtn) {
            triggerBtn.classList.remove('panel-open');
        }
    }

    function getLanguageSwitchGreeting(lang) {
        if (lang === 'ta') {
            return "வணக்கம்! AI DR Screen உதவியாளர் தமிழ் மொழிக்கு மாற்றப்பட்டுள்ளது. நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?";
        } else if (lang === 'hi') {
            return "नमस्ते! AI DR Screen सहायक हिंदी भाषा में सेट हो गया है। मैं आपकी क्या सहायता कर सकता हूँ?";
        }
        return "Language switched to English. How can I help you today?";
    }

    function getCurrentScreeningContext() {
        try {
            const raw = localStorage.getItem('aidr_screening_context');
            if (raw) {
                return JSON.parse(raw);
            }
        } catch (e) {}
        return {};
    }

    function getCurrentPage() {
        const hash = window.location.hash.replace('#', '').trim().toLowerCase();
        return hash || 'pipeline';
    }

    async function handleSendMessage() {
        if (!inputField || isWaitingResponse) return;
        const text = inputField.value.trim();
        if (!text) return;

        inputField.value = '';
        appendUserMessage(text);
        showTypingIndicator();

        const context = getCurrentScreeningContext();
        const page = getCurrentPage();

        try {
            isWaitingResponse = true;
            if (sendBtn) sendBtn.disabled = true;

            const res = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    session_id: sessionId,
                    page: page,
                    language: currentLanguage,
                    context: context
                })
            });

            hideTypingIndicator();
            isWaitingResponse = false;
            if (sendBtn) sendBtn.disabled = false;

            if (res.ok) {
                const data = await res.json();
                const replyText = data.answer || data.message || "I'm ready to help you with AI DR Screen.";
                appendAssistantMessage(replyText, data.sources, data.intent, data.quick_replies);
                executeClientSideActions(data.intent, text, data.actions);
            } else {
                appendAssistantMessage("I'm having trouble connecting right now. Please try again in a moment. You can still use the website's screening features.");
            }
        } catch (err) {
            hideTypingIndicator();
            isWaitingResponse = false;
            if (sendBtn) sendBtn.disabled = false;
            appendAssistantMessage("I'm having trouble connecting right now. Please try again in a moment. You can still use the website's screening features.");
        }
    }

    function handleQuickAction(actionId, label) {
        const queries = {
            start_screening: "How do I start screening a patient?",
            symptom_checker: "What are the common symptoms of diabetic retinopathy and when should I get screened?",
            upload_image: "How do I upload an eye image and what formats are supported?",
            upload_report: "What medical reports can I upload and how does OCR extraction work?",
            understand_result: "What does my current screening result and DR grade mean?",
            explain_ai: "Explain how Grad-CAM, SHAP, and U-Net work together.",
            find_healthcare: "Where can I find a verified healthcare center or eye hospital?",
            health_insights: "What are key health insights and lifestyle choices for preventing diabetic retinopathy progression?",
            navigate: "How do I navigate the website and what keyboard shortcuts exist?",
            how_it_works: "What is AI DR Screen and how does the 5-stage pipeline work?",
            privacy_safety: "How is patient data kept private and what are the medical safety rules?",
            support: "How do I resolve technical issues or contact support?"
        };

        const query = queries[actionId] || label;

        // Directly navigate if action is start_screening or upload
        if (actionId === 'start_screening' || actionId === 'upload_image') {
            if (typeof window.switchPortalTab === 'function') {
                window.switchPortalTab('pipeline', true);
            }
        } else if (actionId === 'navigate') {
            if (typeof window.switchPortalTab === 'function') {
                window.switchPortalTab('dashboard', true);
            }
        }

        if (inputField) {
            inputField.value = query;
            handleSendMessage();
        }
    }

    function executeClientSideActions(intent, text, actions = []) {
        // 1. Structured action execution
        if (actions && Array.isArray(actions) && actions.length > 0) {
            actions.forEach(act => {
                if (act.type === 'navigate' && act.target) {
                    if (typeof window.switchPortalTab === 'function') {
                        window.switchPortalTab(act.target, true);
                    }
                }
            });
            return;
        }

        // 2. Keyword fallback execution
        const lower = text.toLowerCase();
        if (intent === 'navigation' || intent === 'start_screening') {
            if (lower.includes('pipeline') || lower.includes('screening') || lower.includes('upload')) {
                if (typeof window.switchPortalTab === 'function') window.switchPortalTab('pipeline', true);
            } else if (lower.includes('dashboard')) {
                if (typeof window.switchPortalTab === 'function') window.switchPortalTab('dashboard', true);
            } else if (lower.includes('medivision')) {
                if (typeof window.switchPortalTab === 'function') window.switchPortalTab('medivision', true);
            } else if (lower.includes('login')) {
                if (typeof window.switchPortalTab === 'function') window.switchPortalTab('login', true);
            } else if (lower.includes('hero') || lower.includes('home')) {
                if (typeof window.switchPortalTab === 'function') window.switchPortalTab('hero', true);
            }
        }
    }

    function appendUserMessage(text) {
        if (!messagesContainer) return;
        const msgDiv = document.createElement('div');
        msgDiv.className = 'aidr-msg user';
        msgDiv.innerHTML = `
            <div class="aidr-msg-content user-align">
                <div class="aidr-msg-bubble">${escapeHtml(text)}</div>
                <div class="aidr-msg-meta"><span>You</span> • <span>${getCurrentTime()}</span></div>
            </div>
        `;
        messagesContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendAssistantMessage(text, sources = [], intent = '', quick_replies = []) {
        if (!messagesContainer) return;
        const msgDiv = document.createElement('div');
        msgDiv.className = 'aidr-msg assistant';

        let formattedText = formatMarkdown(text);
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            const cleanSources = sources.map(s => s.split('/').pop().replace('.md', '').replace(/_/g, ' ')).slice(0, 2);
            sourcesHtml = cleanSources.map(s => `<span class="aidr-source-badge">📖 ${s}</span>`).join(' ');
        }

        let quickRepliesHtml = '';
        if (quick_replies && Array.isArray(quick_replies) && quick_replies.length > 0) {
            quickRepliesHtml = `
                <div class="aidr-msg-quick-replies">
                    ${quick_replies.map(qr => `<button type="button" class="aidr-reply-pill" data-reply="${escapeHtml(qr)}">${escapeHtml(qr)}</button>`).join('')}
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <img src="oculyn_avatar.png" alt="Oculyn" class="aidr-msg-avatar">
            <div class="aidr-msg-content">
                <div class="aidr-msg-bubble">${formattedText}</div>
                ${quickRepliesHtml}
                <div class="aidr-msg-meta">
                    <span>Oculyn</span> • <span>${getCurrentTime()}</span>
                    ${sourcesHtml ? ` • ${sourcesHtml}` : ''}
                </div>
            </div>
        `;
        messagesContainer.appendChild(msgDiv);

        // Bind click events on quick reply pills
        const pills = msgDiv.querySelectorAll('.aidr-reply-pill');
        pills.forEach(pill => {
            pill.addEventListener('click', (e) => {
                e.preventDefault();
                const replyText = pill.getAttribute('data-reply');
                if (inputField && replyText) {
                    inputField.value = replyText;
                    handleSendMessage();
                }
            });
        });

        scrollToBottom();
    }

    let typingEl = null;
    function showTypingIndicator() {
        if (typingEl || !messagesContainer) return;
        typingEl = document.createElement('div');
        typingEl.className = 'aidr-msg assistant aidr-typing-msg';
        typingEl.innerHTML = `
            <img src="oculyn_avatar.png" alt="Oculyn" class="aidr-msg-avatar">
            <div class="aidr-msg-content">
                <div class="aidr-typing-indicator">
                    <div class="aidr-typing-dot"></div>
                    <div class="aidr-typing-dot"></div>
                    <div class="aidr-typing-dot"></div>
                </div>
            </div>
        `;
        messagesContainer.appendChild(typingEl);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        if (typingEl && typingEl.parentNode) {
            typingEl.parentNode.removeChild(typingEl);
        }
        typingEl = null;
    }

    async function clearChatSession() {
        if (!confirm("Clear this conversation?")) return;
        try {
            await fetch('/api/chat/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
        } catch (e) {}

        if (messagesContainer) {
            messagesContainer.innerHTML = `
                <div class="aidr-msg assistant">
                    <img src="oculyn_avatar.png" alt="Oculyn" class="aidr-msg-avatar">
                    <div class="aidr-msg-content">
                        <div class="aidr-msg-bubble">
                            Hi! 👋 Conversation cleared. How can I assist you with AI DR Screen?
                        </div>
                        <div class="aidr-msg-meta">
                            <span>Oculyn • AI Assistant</span>
                        </div>
                    </div>
                </div>
            `;
        }
    }


    function initContextListeners() {
        // Listen for screening updates from iframes via postMessage or storage
        window.addEventListener('message', (e) => {
            if (e.data && e.data.action === 'screeningUpdated' && e.data.context) {
                localStorage.setItem('aidr_screening_context', JSON.stringify(e.data.context));
            }
        });
    }

    function formatMarkdown(text) {
        if (!text) return '';
        let escaped = escapeHtml(text);
        // Bold
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italics
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Bullet points
        escaped = escaped.replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>');
        escaped = escaped.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
        // Line breaks
        escaped = escaped.replace(/\n\n/g, '<br><br>');
        escaped = escaped.replace(/\n/g, '<br>');
        return escaped;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function getCurrentTime() {
        const d = new Date();
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function scrollToBottom() {
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    // Expose global helper
    window.AIDR_Chatbot = {
        open: toggleChatPanel,
        close: closeChatPanel,
        send: (msg) => {
            if (inputField) {
                inputField.value = msg;
                handleSendMessage();
            }
        }
    };

})();
