// About.js
import React from 'react';
import './About.css';
import Navbar from '../../../../components/nav_bar/navbar';
import Footer from '../../../../components/footer/footer';

const About = () => {
    return (
        <div className="about-container">
            <h1 className="about-title">About Study Peace</h1>
            <p className="about-description">
                Welcome to Study Peace, a serene space designed for individuals seeking
                a calm and peaceful environment to study, meditate, and focus. Our
                mission is to foster mindfulness and balance by providing resources,
                tools, and guidance that promote well-being and mental clarity.
            </p>
            <p className="about-description">
                Whether you are studying for exams, working on a project, or just
                looking to take a moment to breathe and reflect, Study Peace is here to
                support you with a tranquil atmosphere that encourages deep focus and
                relaxation.
            </p>
        </div>
    );
};

export default About;
