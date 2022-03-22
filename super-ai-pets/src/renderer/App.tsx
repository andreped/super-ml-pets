import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import icon from '../../assets/icon.svg';
import './App.css';

const Main = () => {
  return (
    <div>
      <img id="logo" alt="icon" src={icon} />
      <h1 style={{ marginLeft: '5%' }}>Super AI Pets</h1>

      <button id="run" type="button">
        Run AI
      </button>

      <div className="ai"></div>
    </div>
  );
};

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Main />} />
      </Routes>
    </Router>
  );
}
