async function sendMessage() {
    const message = document.getElementById('user-input').value;
    const userId = "user123"; // Replace with a dynamic user ID if needed
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_id: userId, message })
    });
    const data = await response.json();
    document.getElementById('chat-log').innerHTML += `<div class="message user">User: ${message}</div>`;
    document.getElementById('chat-log').innerHTML += `<div class="message bot">Bot: ${data.response}</div>`;
    document.getElementById('user-input').value = '';
    document.getElementById('chat-log').scrollTop = document.getElementById('chat-log').scrollHeight; // Auto-scroll to the bottom
}

// Add event listener for the Enter key
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default action (form submission)
        sendMessage(); // Call the sendMessage function
    }
});