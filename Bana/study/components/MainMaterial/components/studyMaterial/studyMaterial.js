import React from 'react';
import './StudyMaterials.css';

const StudyMaterials = () => {
    const materials = [
        {
            subject: 'Math',
            title: 'Calculus Notes',
            link: '#',
            image: 'https://picsum.photos/seed/picsum/150' // Placeholder image URL
        },
        {
            subject: 'Science',
            title: 'Physics Chapter 1',
            link: '#',
            image: 'https://picsum.photos/seed/picsum/150' // Placeholder image URL
        },
        // Add more materials with images
    ];

    return (
        <div className="study-materials-container">
            <h2 className="study-materials-title">Study Materials</h2>
            <div className="study-materials">
                {materials.map((material, index) => (
                    <div className="material-card" key={index}>
                        <a href={material.link} target="_blank" rel="noopener noreferrer">
                            <img src={material.image} alt={material.title} className="material-image" />
                            <div className="material-info">
                                <h3>{material.title}</h3>
                                <p>{material.subject}</p>
                            </div>
                        </a>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default StudyMaterials;
