/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      gridTemplateColumns: {
        // Add a 32-column grid
        '32': 'repeat(32, minmax(0, 1fr))',
      },
    },
  },
  plugins: [],
}
