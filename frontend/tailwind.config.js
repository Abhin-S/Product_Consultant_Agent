/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#11201f",
        mist: "#f4f6ef",
        ember: "#eb5e28",
        pine: "#2a9d8f",
        sky: "#1d4ed8"
      }
    }
  },
  plugins: []
};