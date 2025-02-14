import React from 'react'

function Navbar() {
  return (
    <div className="navbar bg-white">
    <div className="flex-1">
      <a className="btn btn-ghost text-xl">NoCode AImodel Builder</a>
    </div>
    <div className="flex-none">
      <ul className="menu menu-horizontal px-1">
        
      <li><a>Login</a></li>
      </ul>
    </div>
  </div>
  )
}

export default Navbar