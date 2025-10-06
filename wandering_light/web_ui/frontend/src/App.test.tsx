import React from 'react';
import { render, screen } from '@testing-library/react';
import { ChakraProvider } from '@chakra-ui/react';
import App from './App';

// Mock the GraphEditor component since it has complex dependencies
jest.mock('./components/GraphEditor', () => {
  return function MockGraphEditor() {
    return <div data-testid="graph-editor">Mocked Graph Editor</div>;
  };
});

const ChakraWrapper = ({ children }: { children: React.ReactNode }) => (
  <ChakraProvider>{children}</ChakraProvider>
);

describe('App', () => {
  it('renders the main heading', () => {
    render(
      <ChakraWrapper>
        <App />
      </ChakraWrapper>
    );
    
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    expect(screen.getByText('Wandering Light')).toBeInTheDocument();
  });

  it('renders the GraphEditor component', () => {
    render(
      <ChakraWrapper>
        <App />
      </ChakraWrapper>
    );
    
    expect(screen.getByTestId('graph-editor')).toBeInTheDocument();
  });

  it('has proper layout structure', () => {
    const { container } = render(
      <ChakraWrapper>
        <App />
      </ChakraWrapper>
    );
    
    // Check for main container
    const mainBox = container.firstChild;
    expect(mainBox).toBeInTheDocument();
    
    // Check that there are two direct children (heading and flex container)
    expect((mainBox as HTMLElement).childNodes).toHaveLength(2);
  });

  it('renders heading within Chakra styles wrapper', () => {
    render(
      <ChakraWrapper>
        <App />
      </ChakraWrapper>
    );

    const heading = screen.getByText('Wandering Light');
    expect(heading).toBeInTheDocument();
  });
});
