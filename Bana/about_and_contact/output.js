// join all components to show here
import About from "./components/about/about";
import Contact from "./components/contact/contact";
// import Navbar from "../../components/nav_bar/navbar";
// import Footer from "../../components/footer/footer";
import "./style.css";
const AboutandContact = () => {
    return (
        <div className="aboutandcontact">
          <section className="section-left-about">
                <About />
          </section>
           <section className="section-right-contact">
                <Contact />
           </section>


        </div>
    );
}

export default AboutandContact;
