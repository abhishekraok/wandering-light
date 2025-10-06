// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock WebSocket for tests
(global as any).WebSocket = jest.fn().mockImplementation(() => ({
  close: jest.fn(),
  send: jest.fn(),
  readyState: 1, // OPEN state
  onopen: null,
  onclose: null,
  onmessage: null,
  onerror: null,
}));

// Mock axios for API calls
jest.mock('axios'); 