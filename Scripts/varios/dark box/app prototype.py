import tkinter as tk
import easydamodaran
from easydamodaran import plot_price

new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/varios/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)
current_working_directory = os.getcwd()



# Create the main window
window = tk.Tk()
window.geometry("800x600")
window.title("My GUI Program")
window.configure(bg="#FF0000")  # Set background color of the window to red

# Create a label widget with a message
message = "This is a message for you!"
label = tk.Label(window, text=message, font=("Helvetica", 16), bg="#FF0000", fg="white")
label.place(relx=0.5, rely=0.4, anchor="center")  # Position in the center horizontally, and 40% down vertically

# Define button click event
def button_clicked():
    # Update the label text when the button is clicked
    label.config(text="Button clicked!")

# Create a "Click Me" button
button = tk.Button(window, text="Click Me", command=button_clicked, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=10, pady=5)
button.place(relx=0.5, rely=0.6, anchor="center")  # Position below the label, in the center

# Create a "Close" button to close the window
close_button = tk.Button(window, text="Close", command=window.destroy, font=("Helvetica", 14), bg="#FF6347", fg="white", padx=10, pady=5)
close_button.place(relx=0.5, rely=0.7, anchor="center")  # Position below the "Click Me" button

# Start the event loop
window.mainloop()



ticker='aapl'
starttime = "2024-01-01"
endtime = datetime.datetime.today().strftime('%Y-%m-%d')