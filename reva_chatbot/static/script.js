/**
 * ================================================
 * REVA University Chatbot - Frontend JavaScript
 * ================================================
 */

(function() {
    'use strict';

    // DOM Elements
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');
    const emojiBtn = document.getElementById('emojiBtn');
    const quickSuggestions = document.querySelectorAll('.suggestion-btn');

    // State
    let isTyping = false;
    let isListening = false;
    let conversationHistory = [];

    // Emoji list for quick insertion
    const emojis = ['👋', '🎓', '📚', '🏛️', '📅', '❓', '👍', '🙏'];

    /**
     * Initialize the chatbot
     */
    function init() {
        setupEventListeners();
        messageInput.focus();
        console.log('REVA Chatbot initialized');
    }

    /**
     * Setup all event listeners
     */
    function setupEventListeners() {
        // Form submission
        chatForm.addEventListener('submit', handleFormSubmit);

        // Input events
        messageInput.addEventListener('input', handleInputChange);
        messageInput.addEventListener('keydown', handleKeyDown);

        // Quick suggestion buttons
        quickSuggestions.forEach(btn => {
            btn.addEventListener('click', () => {
                const message = btn.getAttribute('data-message');
                if (message) {
                    sendMessage(message);
                }
            });
        });

        // Voice button
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            voiceBtn.addEventListener('click', toggleVoiceRecognition);
        } else {
            voiceBtn.style.display = 'none';
        }

        // Emoji button
        emojiBtn.addEventListener('click', showEmojiPicker);
    }

    /**
     * Handle form submission
     */
    function handleFormSubmit(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (message && !isTyping) {
            sendMessage(message);
        }
    }

    /**
     * Handle input change events
     */
    function handleInputChange() {
        // Enable/disable send button based on input
        sendBtn.disabled = !messageInput.value.trim();
    }

    /**
     * Handle keyboard events
     */
    function handleKeyDown(e) {
        // Send message on Enter (without Shift)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const message = messageInput.value.trim();
            if (message && !isTyping) {
                sendMessage(message);
            }
        }
    }

    /**
     * Send a message to the chatbot
     */
    async function sendMessage(message) {
        // Add user message to chat
        addMessage(message, 'user');

        // Clear input
        messageInput.value = '';
        sendBtn.disabled = true;

        // Show typing indicator
        showTypingIndicator();

        // Add to conversation history
        conversationHistory.push({ role: 'user', content: message });

        try {
            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            // Hide typing indicator
            hideTypingIndicator();

            // Add bot response
            if (data.status === 'success') {
                addMessage(data.response, 'bot');
                conversationHistory.push({ role: 'bot', content: data.response });
            } else {
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }

        } catch (error) {
            hideTypingIndicator();
            console.error('Error:', error);
            addMessage('Sorry, I\'m having trouble connecting. Please check your internet connection.', 'bot');
        }
    }

    /**
     * Add a message to the chat
     */
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        if (sender === 'bot') {
            avatarDiv.innerHTML = '<img src="/logo" alt="REVA Logo" style="width: 32px; height: 32px; object-fit: contain;">';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';

        // Process text for formatting
        bubbleDiv.innerHTML = formatMessage(text);

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = getCurrentTime();

        contentDiv.appendChild(bubbleDiv);
        contentDiv.appendChild(timeSpan);

        if (sender === 'bot') {
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
        } else {
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(avatarDiv);
        }

        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        scrollToBottom();
    }

    /**
     * Format message text with basic HTML
     */
    function formatMessage(text) {
        // Convert URLs to links
        text = text.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" target="_blank" style="color: inherit; text-decoration: underline;">$1</a>'
        );

        // Convert **bold** to <strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert line breaks to <br>
        text = text.replace(/\n/g, '<br>');

        // Wrap in paragraph
        return `<p>${text}</p>`;
    }

    /**
     * Get current time in HH:MM format
     */
    function getCurrentTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    /**
     * Show typing indicator
     */
    function showTypingIndicator() {
        isTyping = true;
        typingIndicator.style.display = 'flex';
        scrollToBottom();
    }

    /**
     * Hide typing indicator
     */
    function hideTypingIndicator() {
        isTyping = false;
        typingIndicator.style.display = 'none';
    }

    /**
     * Scroll chat to bottom
     */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    /**
     * Toggle voice recognition
     */
    function toggleVoiceRecognition() {
        if (!isListening) {
            startVoiceRecognition();
        } else {
            stopVoiceRecognition();
        }
    }

    /**
     * Start voice recognition
     */
    function startVoiceRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            alert('Voice recognition is not supported in your browser.');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            voiceBtn.classList.add('listening');
            voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            messageInput.value = transcript;
            messageInput.focus();
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            stopVoiceRecognition();
        };

        recognition.onend = () => {
            stopVoiceRecognition();
        };

        recognition.start();
    }

    /**
     * Stop voice recognition
     */
    function stopVoiceRecognition() {
        isListening = false;
        voiceBtn.classList.remove('listening');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    }

    /**
     * Show emoji picker (simple implementation)
     */
    function showEmojiPicker() {
        // Simple emoji insertion - cycles through emojis
        const currentEmoji = messageInput.value.slice(-2);
        const currentIndex = emojis.indexOf(currentEmoji);
        const nextIndex = (currentIndex + 1) % emojis.length;

        if (currentIndex >= 0) {
            messageInput.value = messageInput.value.slice(0, -2) + emojis[nextIndex];
        } else {
            messageInput.value += emojis[0];
        }

        messageInput.focus();
    }

    /**
     * Auto-resize textarea functionality (if using textarea)
     */
    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Expose some functions globally for debugging
    window.REVAChatbot = {
        sendMessage: sendMessage,
        clearHistory: () => {
            conversationHistory = [];
            // Keep only welcome message
            const welcomeMessage = chatMessages.firstElementChild;
            chatMessages.innerHTML = '';
            if (welcomeMessage) {
                chatMessages.appendChild(welcomeMessage);
            }
        },
        getHistory: () => conversationHistory
    };

})();
