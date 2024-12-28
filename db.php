<?php
$servername = "localhost";
$username = "root"; // Replace with your phpMyAdmin username
$password = ""; // Replace with your phpMyAdmin password
$dbname = "user_auth"; // Replace with your database name

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>
