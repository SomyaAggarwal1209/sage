import React from 'react';
import './BookMarketplace.css';

const books = [
    { title: 'Algorithms Unlocked', price: '$29.99', img: 'https://picsum.photos/seed/picsum/150', url: 'https://example.com/algorithms-unlocked' },
    { title: 'Artificial Intelligence: A Modern Approach', price: '$99.99', img: 'https://picsum.photos/seed/picsum/150', url: 'https://example.com/ai-modern-approach' },
    // Add more books with images and links
];

const BookMarketplace = () => {
    return (
        <div className="book-marketplace-container">
            <h2 className="marketplace-title">Book Marketplace</h2>
            <div className="books-grid">
                {books.map((book, index) => (
                    <div key={index} className="book-card">
                        <img src={book.img} alt={book.title} className="book-image" />
                        <div className="book-info">
                            <h3>{book.title}</h3>
                            <p className="book-price">{book.price}</p>
                            <a href={book.url} target="_blank" rel="noopener noreferrer" className="buy-button">Buy Now</a>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default BookMarketplace;
