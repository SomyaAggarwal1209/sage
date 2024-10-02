import React from 'react';
import './VideoLectures.css';

const VideoLectures = () => {
    const lectures = [
        {
            title: 'Lecture 1: Introduction to Algebra',
            link: '#',
            image: 'https://picsum.photos/seed/picsum/150' // Placeholder image URL
        },
        {
            title: 'Lecture 2: Basic Calculus',
            link: '#',
            image: 'https://picsum.photos/seed/picsum/150' // Placeholder image URL
        },
        // Add more lectures with images
    ];

    return (
        <div className="video-lectures-container">
            <h2 className="video-lectures-title">Video Lectures</h2>
            <div className="video-lectures">
                {lectures.map((lecture, index) => (
                    <div className="lecture-card" key={index}>
                        <a href={lecture.link} target="_blank" rel="noopener noreferrer">
                            <img src={lecture.image} alt={lecture.title} className="lecture-image" />
                            <div className="lecture-info">
                                <h3>{lecture.title}</h3>
                            </div>
                        </a>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default VideoLectures;
