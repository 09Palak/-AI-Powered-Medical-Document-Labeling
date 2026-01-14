-- Step 1: Create your database
CREATE DATABASE IF NOT EXISTS medical_data; 

-- Step 2: Use the database
USE medical_data;

-- Step 3: Drop the table if it already exists (useful for development to start fresh)
-- NOTE: In production, remove or comment out the DROP TABLE command.
DROP TABLE IF EXISTS document_analysis;

-- Step 4: Create the table with all necessary columns
CREATE TABLE document_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120),
    email VARCHAR(120),
    phone VARCHAR(20),
    gender VARCHAR(20),
    tag VARCHAR(50),
    imaging_type VARCHAR(50),
    body_part_imaged VARCHAR(100),
    medical_tests TEXT, -- Store as JSON string for lists
    medicines TEXT,     -- Store as JSON string for lists
    doctor_name VARCHAR(100),
    total_bill VARCHAR(50),
    extracted_text LONGTEXT, -- LONGTEXT for potentially very large extracted text
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 5: Describe the table to verify its structure
DESCRIBE document_analysis;
