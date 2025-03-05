const { chromium } = require('playwright');
const { test, expect } = require('@playwright/test');

const fs = require('fs');

const { EMAIL_SELECTOR, PASSWORD_SELECTOR, LOGIN_BUTTON_SELECTOR, URL, JSON_FILE, PROFILE_URL, PROFILE_EMAIL_SELECTOR} = require('../test_data/constans');

test('should login successfully', async ({ page }) => {
  // Launch the browser and navigate to the login page
  const browser = await chromium.launch({ channel: 'chrome', headless: false });
  await page.goto(URL);

  // Read the credentials from the JSON file
  const credentials = await readCredentialsFromFile(JSON_FILE);

  // Enter the email and password
  await page.type(EMAIL_SELECTOR, credentials.email);
  await page.type(PASSWORD_SELECTOR, credentials.password);

  // Click the login button
  await page.click(LOGIN_BUTTON_SELECTOR);

  // Wait for the profile page to load
  await page.waitForURL(PROFILE_URL);

  // Wait for the profile data to be updated
  await page.waitForSelector(PROFILE_EMAIL_SELECTOR);

  // Check that the profile email element is present and has the correct value
  const profileEmail = await page.$eval(PROFILE_EMAIL_SELECTOR, (el) => el.textContent);
  expect(profileEmail).toBe(credentials.email);

  // Close the browser
  await browser.close();
});

async function readCredentialsFromFile(file) {
  const data = await fs.promises.readFile(file, 'utf8');
  return JSON.parse(data);
}