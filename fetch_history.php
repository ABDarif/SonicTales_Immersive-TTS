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

// Fetch the text history from the database
$query = "SELECT id, text_content FROM text_history ORDER BY created_at DESC";
$result = $conn->query($query);

$history = [];
if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $history[] = [
            'id' => $row['id'],
            'text_content' => $row['text_content']
        ];
    }
}

echo json_encode($history);

$conn->close();
?>
