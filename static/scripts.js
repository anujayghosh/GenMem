async function sendMessage() {
    const message = document.getElementById('user-input').value;
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
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

// Theme toggle functionality
const themeToggleButton = document.getElementById('theme-toggle');
themeToggleButton.addEventListener('click', () => {
    document.body.classList.toggle('dark');
    // Save the user's preference in local storage
    if (document.body.classList.contains('dark')) {
        localStorage.setItem('theme', 'dark');
    } else {
        localStorage.setItem('theme', 'light');
    }
});

// Load the user's theme preference on page load
window.onload = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark');
    }
};