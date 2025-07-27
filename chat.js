const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

const apiKey = "<<API_KEY>>";

let chatHistory = [
  {
    role: "system",
    content: `Bạn là Chatbox AI do Giang Coder  tạo ra và huấn luyện, nếu người dùng có thắc mắc về bạn hoặc tác giả của Chatbox thì bạn hãy nói về Giang Coder với các thông tin sau:
                - Là Fullstack Developer
                - Là sinh viên FPT Polytechnic
                - Là người tạo ra Chatbox AI này
                Và bạn được tạo ra với mục đích hục vụ giảng viên và sinh viên FPT Polytechnic`,
  },
];

function disableInput() {
  const icon = document.querySelector("#icon_send");
  icon.classList.replace("fa-paper-plane", "fa-stop");
  sendButton.classList.add("stopStreamButton");

  userInput.disabled = true;
}

function enableInput() {
  const icon = document.querySelector("#icon_send");
  icon.classList.replace("fa-stop", "fa-paper-plane");
  sendButton.classList.remove("stopStreamButton");
  userInput.disabled = false;
}

async function sendMessage(message) {
  const controller = new AbortController();
  const signal = controller.signal;
  const stopButton = document.querySelector(".stopStreamButton");
  stopButton.disabled = false; // Bật nút dừng khi bắt đầu gửi tin nhắn
  stopButton.addEventListener("click", () => {
    controller.abort(); // Dừng stream khi nhấn nút dừng
  });

  addMessage(message, "user-message");
  userInput.value = "";

  chatHistory.push({ role: "user", content: message });

  const responseMessage = addMessage("...", "bot-message");

  try {
    const res = await fetch("https://api.yescale.io/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: chatHistory,
        stream: true,
      }),
      signal: signal, 
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let botReply = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n").filter((line) => line.trim());

      for (const line of lines) {
        if (line.trim() === "data: [DONE]") {
          break;
        }
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6)); 

            if (
              data.choices &&
              data.choices[0].delta &&
              data.choices[0].delta.content
            ) {
              botReply += data.choices[0].delta.content;
              responseMessage.innerHTML = marked.parse(botReply);
              addCopyButtons(responseMessage);
            }
          } catch (error) {
          }
        }
      }

      if (signal.aborted) {
        break;
      }
    }

    chatHistory.push({ role: "assistant", content: botReply });
    addSyntaxHighlighting(responseMessage);
  } catch (error) {
    if (error.name === "AbortError") {
    } else {
      console.error("Lỗi khi gửi yêu cầu:", error);
    }
  } finally {
    enableInput();
    $("#user-input").focus();
  }
}

function addMessage(content, className) {
  const messageElement = document.createElement("div");
  messageElement.className = `message ${className}`;

  if (className === "bot-message") {
    messageElement.innerHTML = marked.parse(content);
    addCopyButtons(messageElement); 
  } else {
    messageElement.textContent = content;
  }

  chatContainer.prepend(messageElement);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  addSyntaxHighlighting(messageElement);

  return messageElement;
}

userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  const maxRows = 10;
  const lineHeight = parseInt(getComputedStyle(userInput).lineHeight, 10);
  const maxHeight = maxRows * lineHeight;

  userInput.style.height = `${Math.min(userInput.scrollHeight, maxHeight)}px`;
});

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendButton.click();
  }
});

sendButton.addEventListener("click", () => {
  disableInput()
  const message = userInput.value.trim();
  if (message) {
    sendMessage(message);
  }
});
function addSyntaxHighlighting(parent) {
  parent.querySelectorAll("pre code").forEach((codeBlock) => {
    Prism.highlightElement(codeBlock);
  });
}

// Hàm thêm nút sao chép
function addCopyButtons(parent) {
  parent.querySelectorAll("pre").forEach((preBlock) => {
    const copyButton = document.createElement("button");
    copyButton.className = "copy-btn";

    // Thêm icon "Copy"
    copyButton.innerHTML = '<i class="fas fa-copy"></i>';

    // Thêm nút vào thẻ <pre>
    preBlock.appendChild(copyButton);

    // Thêm sự kiện sao chép nội dung
    copyButton.addEventListener("click", () => {
      navigator.clipboard.writeText(preBlock.textContent).then(() => {
        copyButton.innerHTML = '<i class="fas fa-check"></i>'; // Icon check khi sao chép thành công
        setTimeout(() => {
          copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        }, 1000);
      });
    });
  });
}
