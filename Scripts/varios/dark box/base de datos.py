import mysql.connector
import sqlite3
import os

# Get the current working directory
current_working_directory = os.getcwd()

print(current_working_directory)

new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/varios/dark box'  # Replace with the path to your desired directory

# Change the current working directory
os.chdir(new_directory)
current_working_directory = os.getcwd()

print(current_working_directory)

# Step 1: Connect to MySQL db
conn = mysql.connector.connect(
    host='localhost',      # Replace with your host
    user='your_username',  # Replace with your MySQL username
    password='your_password',  # Replace with your MySQL password
    database='your_database'   # Replace with your db name
)
conn = sqlite3.connect('example.db')


# Step 2: Create a cursor object to interact with the db
cursor = conn.cursor()

# Step 3: Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Clients (
    CustomerID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100) UNIQUE,
    DateJoined DATE
)
''')

# Step 4: Insert rows into the table
customers = [
    (1, 'John', 'Doe', 'john.doe@example.com', '2023-09-05'),
    (2, 'Jane', 'Smith', 'jane.smith@example.com', '2023-09-06'),
    (3, 'Alice', 'Brown', 'alice.brown@example.com', '2023-09-07')
]

cursor.executemany('''
INSERT INTO Customers (CustomerID, FirstName, LastName, Email, DateJoined)
VALUES (?, ?, ?, ?, ?)
''', customers)

# Step 5: Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()

print("Table created and rows inserted successfully!")
