const { chromium } = require('playwright');
const { test, expect } = require('@playwright/test');

const fs = require('fs');
const faker = require('faker');

const { EMAIL_SELECTOR, PASSWORD_SELECTOR, LOGIN_BUTTON_SELECTOR, URL, JSON_FILE, PROFILE_URL, PROFILE_EMAIL_SELECTOR, PACIENT_LI_SELECTOR, PACIENT_LI_ANCHOR_SELECTOR, CREATE_ACCOUNT_BUTTON_SELECTOR, FORM_NAME_SELECTOR , FORM_EMAIL_SELECTOR, FORM_PASSWORD_SELECTOR, FORM_ADDRESS_SELECTOR, FORM_ABOUT_SELECTOR , FORM_PHONE_NUMBER_SELECTOR , FORM_BIRTH_DATE_SELECTOR , FORM_GENDER_SELECTOR , FORM_ASSISTENT_SELECTOR, ADD_ACCOUNT_BUTTON_SELECTOR , NEW_ACCOUNT_VALUES_FILE, SEARCH_NAME_INPUT_SELECTOR, LOGOUT_BUTTON_SELECTOR } = require('../test_data/constans');
const newAccountValues = require(NEW_ACCOUNT_VALUES_FILE);

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

  // Verify that the #pacient-li element is present and accessible
  await page.waitForSelector(PACIENT_LI_SELECTOR);
  expect(await page.isVisible(PACIENT_LI_SELECTOR)).toBe(true);

  // Access the #pacient-li > a element
  const pacientLiAnchor = await page.$(PACIENT_LI_ANCHOR_SELECTOR);
  expect(pacientLiAnchor).not.toBeNull();

  const link = await page.$('#link-pacients');
  await link.click();

  const button = await page.$(CREATE_ACCOUNT_BUTTON_SELECTOR);
  await button.click();

  await page.type(FORM_PASSWORD_SELECTOR, newAccountValues.password);
  await page.type(FORM_ADDRESS_SELECTOR, newAccountValues.address);
  await page.type(FORM_ABOUT_SELECTOR, newAccountValues.about);
  await page.type(FORM_PHONE_NUMBER_SELECTOR, newAccountValues.phone);
  await page.type(FORM_BIRTH_DATE_SELECTOR, newAccountValues.birth_date);

  const firstName = faker.name.firstName();
  const lastName = faker.name.lastName();
  const randomName = firstName + ' ' + lastName;
  const randomEmail = firstName + '_' + lastName + '@gmail.com';

  await page.type(FORM_NAME_SELECTOR, randomName);
  await page.type(FORM_EMAIL_SELECTOR, randomEmail);

  await page.check(FORM_GENDER_SELECTOR);
  await page.check(FORM_ASSISTENT_SELECTOR);

  const addAccountButton = await page.$(ADD_ACCOUNT_BUTTON_SELECTOR);
  await addAccountButton.click();

  // Enter the randomName value in the search_name input field
  await page.type(SEARCH_NAME_INPUT_SELECTOR, randomName);

  // Wait for the search results to be updated
  await page.waitForSelector(`.pacients-container tbody tr`);

  // Get all the email cells in the table
  const emailCells = await page.$$eval(`.pacients-container tbody tr td:nth-child(3)`, (cells) => cells.map((cell) => cell.textContent));

  // Check if the generated email is present in the list of email cells
  expect(emailCells).toContain(randomEmail);

  const logoutButton = await page.$(LOGOUT_BUTTON_SELECTOR);
  await logoutButton.click();

  await page.waitForURL(URL);

  // Close the browser
  await browser.close();
});

async function readCredentialsFromFile(file) {
  const data = await fs.promises.readFile(file, 'utf8');
  return JSON.parse(data);
}