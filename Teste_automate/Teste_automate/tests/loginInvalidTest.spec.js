const { chromium } = require('playwright');
const { test, expect } = require('@playwright/test');

const fs = require('fs');

const { EMAIL_SELECTOR, PASSWORD_SELECTOR, LOGIN_BUTTON_SELECTOR, URL, JSON_FILE_INVALID, TOAST_MESSAGE_SELECTOR} = require('../test_data/constans');


const JSON_FILE = 'test_data/invalid_credentials.json';


test('should display toast message with "Invalid credentials" when entering invalid credentials', async ({ page }) => {
    // Launch the browser and navigate to the login page
    const browser = await chromium.launch({ channel: 'chrome', headless: false });
    await page.goto(URL);
  
    // Read the credentials from the JSON file
    const credentials = await readCredentialsFromFile(JSON_FILE_INVALID);
  
    // Enter the email and password
    await page.type(EMAIL_SELECTOR, credentials.email);
    await page.type(PASSWORD_SELECTOR, credentials.password);
  
    // Click the login button
    await page.click(LOGIN_BUTTON_SELECTOR);

    // Wait for the profile data to be updated
    await page.waitForSelector(TOAST_MESSAGE_SELECTOR);
  
    // Check that the toast message contains the text "Invalid credentials"
    const toastMessage = await page.$eval(TOAST_MESSAGE_SELECTOR, (el) => el.textContent);
    expect(toastMessage).toContain('Invalid credentials');
    
    // Wait for the toast message to disappear
    await page.waitForSelector(TOAST_MESSAGE_SELECTOR, { state: 'hidden' });

    // Verify that the email and password fields are cleared
    expect(await page.inputValue(EMAIL_SELECTOR)).toBe('');
    expect(await page.inputValue(PASSWORD_SELECTOR)).toBe('');
  
    // Close the browser
    await browser.close();
});

async function readCredentialsFromFile(file) {
  const data = await fs.promises.readFile(file, 'utf8');
  return JSON.parse(data);
}