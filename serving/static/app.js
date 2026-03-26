const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant developed by Koma Labs.";

const state = {
  messages: [],
  draft: "",
  isLoading: false,
  error: null,
};

const elements = {
  body: document.body,
  emptyState: document.getElementById("empty-state"),
  chatView: document.getElementById("chat-view"),
  chatScroll: document.getElementById("chat-scroll"),
  chatMessages: document.getElementById("chat-messages"),
  input: document.getElementById("message-input"),
  sendButton: document.getElementById("send-button"),
  newChatButton: document.getElementById("new-chat-button"),
  errorBanner: document.getElementById("error-banner"),
  suggestionCards: Array.from(document.querySelectorAll("[data-prompt]")),
};

let activeController = null;
let requestVersion = 0;

function formatTimestamp(date = new Date()) {
  return new Intl.DateTimeFormat([], {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function autoResizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 240)}px`;
  elements.input.style.overflowY = elements.input.scrollHeight > 240 ? "auto" : "hidden";
}

function syncComposer() {
  if (elements.input.value !== state.draft) {
    elements.input.value = state.draft;
  }

  autoResizeInput();
  elements.sendButton.disabled = state.isLoading || !state.draft.trim();
  elements.sendButton.setAttribute("aria-busy", String(state.isLoading));
}

function syncError() {
  if (state.error) {
    elements.errorBanner.hidden = false;
    elements.errorBanner.textContent = state.error;
    return;
  }

  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function createMessageElement(message) {
  const messageElement = document.createElement("article");
  messageElement.className = `message message--${message.role}`;

  const meta = document.createElement("div");
  meta.className = "message__meta";

  const label = document.createElement("span");
  label.textContent = message.role === "assistant" ? "GPT 2.5" : "You";
  meta.appendChild(label);

  const bubble = document.createElement("div");
  bubble.className = "message__bubble";
  bubble.textContent = message.content;

  const timestamp = document.createElement("span");
  timestamp.className = "message__timestamp";
  timestamp.textContent = message.timestamp;

  messageElement.append(meta, bubble, timestamp);
  return messageElement;
}

function createTypingIndicator() {
  const typingElement = document.createElement("div");
  typingElement.className = "typing";

  const dots = document.createElement("div");
  dots.className = "typing__dots";
  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.className = "typing__dot";
    dots.appendChild(dot);
  }

  const label = document.createElement("span");
  label.className = "typing__label";
  label.textContent = "Generating...";

  typingElement.append(dots, label);
  return typingElement;
}

function renderMessages() {
  elements.chatMessages.replaceChildren();

  if (state.messages.length === 0) {
    return;
  }

  const separator = document.createElement("div");
  separator.className = "date-separator";
  const separatorLabel = document.createElement("span");
  separatorLabel.textContent = "Today";
  separator.appendChild(separatorLabel);
  elements.chatMessages.appendChild(separator);

  state.messages.forEach((message) => {
    elements.chatMessages.appendChild(createMessageElement(message));
  });

  if (state.isLoading) {
    elements.chatMessages.appendChild(createTypingIndicator());
  }
}

function scrollToBottom() {
  elements.chatScroll.scrollTo({
    top: elements.chatScroll.scrollHeight,
    behavior: "smooth",
  });
}

function render(options = {}) {
  const { scroll = false } = options;
  const hasMessages = state.messages.length > 0;

  elements.body.classList.toggle("chat-active", hasMessages);
  elements.emptyState.hidden = hasMessages;
  elements.chatView.hidden = !hasMessages;

  renderMessages();
  syncComposer();
  syncError();

  if (scroll && hasMessages) {
    requestAnimationFrame(scrollToBottom);
  }
}

function getErrorMessage(response, payload) {
  if (response.status === 503) {
    return "The model is still loading. Please wait a moment and try again.";
  }

  if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }

  return "Something went wrong while contacting GPT 2.5. Please try again.";
}

async function sendMessage(promptOverride) {
  const prompt = (promptOverride ?? state.draft).trim();
  if (!prompt || state.isLoading) {
    return;
  }

  state.error = null;
  state.draft = "";
  state.isLoading = true;
  state.messages = [
    ...state.messages,
    {
      role: "user",
      content: prompt,
      timestamp: formatTimestamp(),
    },
  ];
  render({ scroll: true });

  const requestId = requestVersion + 1;
  requestVersion = requestId;
  activeController = new AbortController();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [
          {
            role: "system",
            content: DEFAULT_SYSTEM_PROMPT,
          },
          ...state.messages.map(({ role, content }) => ({ role, content })),
        ],
      }),
      signal: activeController.signal,
    });

    if (requestId !== requestVersion) {
      return;
    }

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => null);
      state.error = getErrorMessage(response, errorPayload);
      state.isLoading = false;
      activeController = null;
      render({ scroll: true });
      return;
    }

    const payload = await response.json();
    state.messages = [
      ...state.messages,
      {
        role: "assistant",
        content: payload.response?.trim() || "(No response returned.)",
        timestamp: formatTimestamp(),
      },
    ];
    state.isLoading = false;
    activeController = null;
    render({ scroll: true });
  } catch (error) {
    if (activeController?.signal.aborted || requestId !== requestVersion) {
      return;
    }

    state.error = "Unable to reach GPT 2.5 right now. Check the server and try again.";
    state.isLoading = false;
    activeController = null;
    render({ scroll: true });
  }
}

function resetChat() {
  requestVersion += 1;
  if (activeController) {
    activeController.abort();
    activeController = null;
  }

  state.messages = [];
  state.draft = "";
  state.isLoading = false;
  state.error = null;
  render();
  elements.input.focus();
}

elements.input.addEventListener("input", (event) => {
  state.draft = event.target.value;
  syncComposer();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void sendMessage();
  }
});

elements.sendButton.addEventListener("click", () => {
  void sendMessage();
});

elements.newChatButton.addEventListener("click", () => {
  resetChat();
});

elements.suggestionCards.forEach((card) => {
  card.addEventListener("click", () => {
    void sendMessage(card.dataset.prompt ?? "");
  });
});

render();
elements.input.focus();
