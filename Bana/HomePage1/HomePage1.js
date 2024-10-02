import React from 'react';
import { Link } from 'react-router-dom';
// import Navbar from '../../components/nav_bar/navbar';
// import Footer from '../../components/footer/footer';
import './HomePage1.css';

const HomePage1 = () => {
    return (
        <div className="homepage">
            {/* <Navbar /> */}
            <div className="homepage-content">
                <h1>Welcome to Study Peace</h1>
                <p>Your ultimate hub for focused learning and collaboration.</p>

                <div className="features">
                    <div className="feature-box">
                        <h3>Study</h3>
                        <p>Explore our comprehensive study materials, video lectures, test series, note-taking tools, and more!</p>
                        <Link to="/study-page">Go to Study</Link>
                    </div>

                    <div className="feature-box">
                        <h3>Join a Chat</h3>
                        <p>Create study groups, chat individually, or engage in video/audio discussions with your peers.</p>
                        <Link to="/chat-page">Go to Chat </Link>
                    </div>

                    <div className="feature-box">
                        <h3>Games</h3>
                        <p>Boost your knowledge while having fun with our interactive study games.</p>
                        <Link to="/study-games-page">Go to Games </Link>
                    </div>


                </div>
            </div>
            {/* <Footer /> */}
        </div>
    );
};

export default HomePage1;
