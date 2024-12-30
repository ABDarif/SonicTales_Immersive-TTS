<?php
header('Content-Type: application/json');

// Database configuration
$host = 'localhost';
$username = 'root';
$password = ''; // Replace with your database password
$database = 'sonictales';

// Connect to the database
$conn = new mysqli($host, $username, $password, $database);

if ($conn->connect_error) {
    echo json_encode(['success' => false, 'error' => 'Database connection failed.']);
    exit;
}

// Retrieve the JSON data from the request
$data = json_decode(file_get_contents('php://input'), true);

if (!isset($data['text']) || trim($data['text']) === '') {
    echo json_encode(['success' => false, 'error' => 'Text input is empty.']);
    exit;
}

$text = $conn->real_escape_string($data['text']);

// Insert the text into the database
$query = "INSERT INTO text_history (text_content) VALUES ('$text')";

if ($conn->query($query) === TRUE) {
    echo json_encode(['success' => true]);
} else {
    echo json_encode(['success' => false, 'error' => 'Failed to save text.']);
}

$conn->close();
?>
