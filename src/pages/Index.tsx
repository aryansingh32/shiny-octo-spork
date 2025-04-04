
import React from "react";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate("/");
  }, [navigate]);

  return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
};

export default Index;
