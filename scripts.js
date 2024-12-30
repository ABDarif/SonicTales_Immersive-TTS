document.getElementById('playButton').addEventListener('click', () => {
    const textInput = document.getElementById('textInput').value;
    const voice = document.getElementById('voiceSelect').value;
    const mood = document.getElementById('moodSelect').value;
    const output = document.getElementById('output');

    if (!textInput.trim()) {
        alert("Please enter some text.");
        return;
    }

    output.classList.remove('hidden');
    output.innerHTML = `<p>Simulating voice: <strong>${voice}</strong> with mood: <strong>${mood}</strong></p>`;

    // Simulated delay to mimic processing
    setTimeout(() => {
        output.innerHTML += `<p>Narrating: "${textInput}"</p>`;
    }, 2000);

    // Save text to the database
    fetch('save_text.php', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: textInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            loadHistory(); // Refresh the history sidebar
        } else {
            alert("Failed to save text.");
        }
    })
    .catch(error => console.error("Error saving text:", error));
});

function loadHistory() {
    fetch('fetch_history.php')
        .then(response => response.json())
        .then(data => {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = '';

            data.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item.text_content;
                historyList.appendChild(li);
            });
        })
        .catch(error => console.error("Error fetching history:", error));
}

// Load history on page load
window.addEventListener('DOMContentLoaded', loadHistory);

document.getElementById('saveButton').addEventListener('click', () => {
    const textInput = document.getElementById('textInput').value;

    if (!textInput.trim()) {
        alert("Please enter some text to save.");
        return;
    }

    fetch('save_text.php', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: textInput })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Text saved successfully!");
            loadHistory(); // Refresh the history sidebar
        } else {
            alert(`Error: ${data.error || 'Failed to save text.'}`);
        }
    })
    .catch(error => console.error("Error:", error));
});
