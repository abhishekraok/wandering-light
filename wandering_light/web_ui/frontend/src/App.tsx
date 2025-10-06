import React from 'react';
import { Box, Flex, Heading, Button, HStack } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import GraphEditor from './components/GraphEditor';
import SolverPlayground from './components/SolverPlayground';

function Navigation() {
  const location = useLocation();
  
  return (
    <Box bg="blue.700" p={4}>
      <Flex justify="space-between" align="center">
        <Heading as="h1" color="white" size="lg">
          Wandering Light
        </Heading>
        <HStack spacing={4}>
          <Button
            as={Link}
            to="/"
            colorScheme="blue"
            variant={location.pathname === '/' ? 'solid' : 'ghost'}
            color={location.pathname === '/' ? 'blue.700' : 'white'}
            bg={location.pathname === '/' ? 'white' : 'transparent'}
          >
            Graph Editor
          </Button>
          <Button
            as={Link}
            to="/solver"
            colorScheme="blue"
            variant={location.pathname === '/solver' ? 'solid' : 'ghost'}
            color={location.pathname === '/solver' ? 'blue.700' : 'white'}
            bg={location.pathname === '/solver' ? 'white' : 'transparent'}
          >
            Solver Playground
          </Button>
        </HStack>
      </Flex>
    </Box>
  );
}

function App() {
  return (
    <Router>
      <Box>
        <Navigation />
        <Routes>
          <Route path="/" element={
            <Flex h="calc(100vh - 80px)">
              <GraphEditor />
            </Flex>
          } />
          <Route path="/solver" element={<SolverPlayground />} />
        </Routes>
      </Box>
    </Router>
  );
}

export default App;