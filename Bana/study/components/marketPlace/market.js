import React, { useState } from 'react';
import CourseMarketplace from "./Components/course_marketPlace/course_market";
import BookMarketplace from "./Components/BookMarketPlace/bookMarketPlace";
import './Marketplace.css';

const ChooseMarketplace = () => {
    const [activeMarketplace, setActiveMarketplace] = useState('');

    const renderMarketplace = () => {
        switch (activeMarketplace) {
            case 'course':
                return <CourseMarketplace />;
            case 'book':
                return <BookMarketplace />;
            default:
                return null; // No marketplace selected
        }
    };

    return (
        <div className="choose-marketplace-container">
            <h1 className="marketplace-title">Choose Your Marketplace</h1>
            <div className="marketplace-options">
                <div
                    className={`marketplace-option ${activeMarketplace === 'book' ? 'selected' : ''}`}
                    onClick={() => setActiveMarketplace('book')}
                >
                    <h2>Book Marketplace</h2>
                    <p>Explore a variety of books available for purchase.</p>
                </div>
                <div
                    className={`marketplace-option ${activeMarketplace === 'course' ? 'selected' : ''}`}
                    onClick={() => setActiveMarketplace('course')}
                >
                    <h2>Course Marketplace</h2>
                    <p>Join our wide range of interactive courses.</p>
                </div>
            </div>
            <div className="selected-marketplace-content">
                {renderMarketplace()}
            </div>
        </div>
    );
};

export default ChooseMarketplace;
