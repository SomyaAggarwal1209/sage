// SearchBar.js
import React, { useState } from 'react';
import './SearchBar.css';

const SearchBar = () => {
    const [searchQuery, setSearchQuery] = useState('');

    const handleSearch = (e) => {
        setSearchQuery(e.target.value);
    };

    return (
        <div className="search-bar">
            <input
                type="text"
                placeholder="Search study materials, subjects..."
                value={searchQuery}
                onChange={handleSearch}
            />
        </div>
    );
};

export default SearchBar;
