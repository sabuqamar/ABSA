import sys
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# default path to file to store data
path_to_file = "reviews2.csv"

# default number of scraped pages
num_page = 4

# default tripadvisor website of restaurant
url = "https://www.tripadvisor.com/Restaurant_Review-g60763-d802686-Reviews-Hard_Rock_Cafe-New_York_City_New_York.html"
url = "https://www.tripadvisor.co.uk/Restaurant_Review-g580426-d1673004-Reviews-The_Rake_Mediterranean_Tapas_Restaurant-Rochdale_Greater_Manchester_England.html"

# if you pass the inputs in the command line
if (len(sys.argv) == 4):
    path_to_file = sys.argv[1]
    num_page = int(sys.argv[2])
    url = sys.argv[3]

# Import the webdriver
driver = webdriver.Chrome()
driver.get(url)
time.sleep(3)
driver.find_element(By.XPATH, './/button[@id="onetrust-accept-btn-handler"]').click()


# Open the file to save the review
csvFile = open(path_to_file, 'w', newline='', encoding="utf-8")
csvWriter = csv.writer(csvFile)

# change the value inside the range to save more or less reviews
for i in range(0, num_page):

    # expand the review 
    time.sleep(2)
    driver.find_element(By.XPATH, "//span[@class='taLnk ulBlueLinks']").click()

    driver.execute_script("window.scrollTo(0, 1000);")
    time.sleep(2)
    container = driver.find_elements(By.XPATH, ".//div[@class='review-container']")
    
    for j in range(len(container)):

        rating = container[j].find_element(By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        review = container[j].find_element(By.XPATH, ".//p[@class='partial_entry']").text.replace("\n", " ")

        csvWriter.writerow([rating, review]) 

    # change the page
    driver.find_element(By.XPATH, './/a[@class="nav next ui_button primary"]').click()

driver.close()