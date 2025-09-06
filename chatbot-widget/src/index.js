import React from "react";
import ReactDOM from "react-dom/client";
import ChatWidget from "./ChatWidget";
import "./ChatWidget.css";

// Look for a div with id 'chatbot-root' (add this in your vanilla HTML)
const container = document.getElementById("chatbot-root");
if (container) {
  const root = ReactDOM.createRoot(container);
  root.render(<ChatWidget apiUrl="http://localhost:5000/api" />);
}
