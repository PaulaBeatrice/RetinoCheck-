module.exports = {
    preset: 'jest/presets/node-without-callbacks',
    testEnvironment: 'node',
    testMatch: ['**/+(*.)+(spec|test).+(ts|js)?(x)'],
    transform: {
      '^.+\\.(ts|js|html)$': 'ts-jest',
    },
  };