import React from 'react'

function Navbar() {
  return (
    <header className="flex items-center justify-between p-6 bg-[#f8f8f8]">
    <div className="flex space-x-2">
      <div className="h-2 w-2 rounded-full bg-black"></div>
      <div className="h-2 w-2 rounded-full bg-black"></div>
    </div>
    <div className="flex items-center space-x-6">
      <button className="text-sm">EN</button>
      <button
       
        className="text-sm hover:underline"
      >
        CONTACT US
      </button>
      <button className="flex flex-col space-y-1">
        <span className="h-0.5 w-6 bg-black"></span>
        <span className="h-0.5 w-6 bg-black"></span>
      </button>
    </div>
  </header>
  )
}

export default Navbar