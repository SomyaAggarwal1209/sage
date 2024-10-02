import { BrowserRouter as Router, Routes, Route, Navigate} from 'react-router-dom';
import HomePage1 from './Bana/HomePage1/HomePage1';
import ChatPage from './Bana/chatPage_from_person/chatpage';
import Chatbot from './components/chatbot/chatbot';
import StudyPage from './Bana/study/studyPage';
import VideoCall from './Bana/chatPage_from_person/components/videoCall/video_call';
import Room from './Bana/chatPage_from_person/components/videoCall/room';
import StudyGamesPage from './Bana/game/PageOfGame';
import AboutandContact from './Bana/about_and_contact/output';
import Navbar from "./components/nav_bar/navbar";
import Footer from "./components/footer/footer";

const App = () => {
    return (
        <>
            <Navbar />
        <Router>
            <Routes>
                <Route path="/" element={<HomePage1 />} />
                <Route path="/study-page" element={<StudyPage />} />
                <Route path="/chat-page" element={<ChatPage />} />
                <Route path="/Bot" element={<Chatbot />}/>
                <Route path="/study-games-page" element={<StudyGamesPage/>}/>
                    <Route path="/Vid" element={<VideoCall />} />
                    <Route path="/room/:roomId" element={<Room />} />
                <Route path="/about" element={<AboutandContact/>}/>
            </Routes>
        </Router>
        <Chatbot/>
            <Footer />
        </>
    );
};

export default App;
