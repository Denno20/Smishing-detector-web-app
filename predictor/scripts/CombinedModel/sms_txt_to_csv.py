
import csv # Library to help write csv files
def main():
    rowList = [["Label", "Text"]] #These columns will appear at the top of the file
    #Open the file
    messages = open("Datasets/sms.txt", "r").readlines()

    #For each message, split the category and text by the tab space
    #Append the row to the list
    for message in messages:
        current = message.strip().split("\t")
        rowList.append(current)

    #Write all the data to a new file
    with open("Datasets/sms_validation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rowList)

if __name__ == "__main__":
    main()

