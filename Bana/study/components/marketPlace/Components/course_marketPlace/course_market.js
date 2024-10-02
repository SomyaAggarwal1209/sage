import React from 'react';
import './CourseMarketplace.css';

const courses = [
    { name: 'React Mastery', price: '$49.99', img: 'https://picsum.photos/seed/picsum/150', url: 'https://example.com/react-mastery' },
    { name: 'Advanced Data Science', price: '$79.99', img: 'https://picsum.photos/seed/picsum/150', url: 'https://example.com/advanced-data-science' },
    // Add more courses with images and links
];

const CourseMarketplace = () => {
    return (
        <div className="course-marketplace-container">
            <h2 className="marketplace-title">Course Marketplace</h2>
            <div className="courses-grid">
                {courses.map((course, index) => (
                    <div key={index} className="course-card">
                        <img src={course.img} alt={course.name} className="course-image" />
                        <div className="course-info">
                            <h3>{course.name}</h3>
                            <p className="course-price">{course.price}</p>
                            <a href={course.url} target="_blank" rel="noopener noreferrer" className="buy-button">Buy Now</a>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default CourseMarketplace;
