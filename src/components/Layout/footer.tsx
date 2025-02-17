import React from 'react'

function Footer() {
  return (
    <footer className="footer footer-center bg-[#f8f8f8] text-primary-content p-10">
  <aside>
    <div className='flex gap-2'>
  <div className="h-2 w-2 rounded-full bg-black"></div>
  <div className="h-2 w-2 rounded-full bg-black"></div>
  </div>
    <p className="font-bold">
      NoCode AIModel Builder
      
  
    </p>
    <p>Copyright © {new Date().getFullYear()} - All right reserved</p>
  </aside>
  <nav>
    <div className="grid grid-flow-col gap-4">
     
    </div>
  </nav>
</footer>
  )
}

export default Footer