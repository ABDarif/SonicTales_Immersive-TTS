
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
});
