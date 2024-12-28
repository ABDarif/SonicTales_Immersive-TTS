
-- SQL script to set up the database and users table for the login/signup system

CREATE DATABASE IF NOT EXISTS sonictales;

USE sonictales;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    password VARCHAR(255) NOT NULL
);
