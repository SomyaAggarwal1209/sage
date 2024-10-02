import CourseMarketplace from "./Components/course_marketPlace/course_market";
import BookMarketplace from "./Components/BookMarketPlace/bookMarketPlace";
import ChooseBarMarketElement from "./Components/BookMarketPlace/bookMarketPlace";

const MarketPlace= () => {
    return (
        <div className="marketplace">
            <ChooseBarMarketElement />
            <BookMarketplace/>
            <CourseMarketplace/>

        </div>
    );
}

export default MarketPlace;
