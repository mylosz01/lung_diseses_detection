﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.EntityFrameworkCore;
using LungMed.Data;
using LungMed.Models;
using LungMed.ViewModels;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Authorization;

namespace LungMed.Controllers
{
    //[Authorize(Roles = "Administrator")]
    public class PatientsController : Controller
    {
        private readonly ApplicationDbContext _context;
        private readonly UserManager<ApplicationUser> _userManager;



        public PatientsController(ApplicationDbContext context, UserManager<ApplicationUser> userManager)
        {
            _context = context;
            _userManager = userManager;


        }

        // GET: Patients
        public async Task<IActionResult> Index(string PersonalNumberPatientSearch, string LastNameDoctorSearch, int DoctorIdSearch)
        {

            var user = await _userManager.GetUserAsync(User);
 
            IQueryable<Patient> patients;

            if (User.IsInRole("Administrator"))
            {
                patients = _context.Patient.Include(p => p.Doctor);
            }
            else
            {
                patients = _context.Patient
                    .Include(p => p.Doctor)
                    .Where(p => p.DoctorId == user.DoctorId);
            }

            if (!string.IsNullOrEmpty(PersonalNumberPatientSearch))
            {
                patients = patients.Where(s => s.PersonalNumber.Contains(PersonalNumberPatientSearch));
            }
            if (!string.IsNullOrEmpty(LastNameDoctorSearch))
            {
                patients = patients.Where(s => s.Doctor.LastName.Contains(LastNameDoctorSearch));
            }
            if (DoctorIdSearch != 0)
            {
                patients = patients.Where(s => s.DoctorId == DoctorIdSearch);
            }

            var patientDoctorModel = new PatientViewModel
            {
                Patients = await patients.ToListAsync()
            };

            return View(patientDoctorModel);
        }

        // GET: Patients/Details/5
        public async Task<IActionResult> Details(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }

            var patient = await _context.Patient
                .Include(p => p.Doctor)
                .FirstOrDefaultAsync(m => m.Id == id);

            

            if (patient == null)
            {
                return NotFound();
            }

            return View(patient);
        }

        // GET: Patients/Create
        public IActionResult Create()
        {
            ViewData["DoctorId"] = new SelectList(_context.Doctor, "Id", "FullNameWithId");
            return View();
        }

        // POST: Patients/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Create([Bind("Id,FirstName,LastName,PersonalNumber,PhoneNumber,BirhtDate,Sex,DoctorId")] Patient patient)
        {
            var existingPatient = await _context.Patient
                .FirstOrDefaultAsync(p => p.PersonalNumber == patient.PersonalNumber);

            if (existingPatient != null)
            {
                ModelState.AddModelError("PersonalNumber", "Personal number must be unique.");
            }

            var existingPhoneNumber = await _context.Patient
                .FirstOrDefaultAsync(p => p.PhoneNumber == patient.PhoneNumber);

            if (existingPhoneNumber != null)
            {
                ModelState.AddModelError("PhoneNumber", "Phone number must be unique.");
            }

            if (ModelState.IsValid)
            {
                _context.Add(patient);
                await _context.SaveChangesAsync();
                return RedirectToAction(nameof(Index));
            }

            ViewData["DoctorId"] = new SelectList(_context.Doctor, "Id", "FullNameWithId", patient.DoctorId);
            return View(patient);
        }


        // GET: Patients/Edit/5
        public async Task<IActionResult> Edit(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }


            var patient = await _context.Patient.FindAsync(id);
            if (patient == null)
            {
                return NotFound();
            }
            ViewData["DoctorId"] = new SelectList(_context.Doctor, "Id", "FullNameWithId", patient.DoctorId);
            return View(patient);
        }

        // POST: Patients/Edit/5
        // To protect from overposting attacks, enable the specific properties you want to bind to.
        // For more details, see http://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Edit(int id, [Bind("Id,FirstName,LastName,PersonalNumber,PhoneNumber,BirhtDate,Sex,DoctorId")] Patient patient)
        {
            if (id != patient.Id)
            {
                return NotFound();
            }

            var existingPatient = await _context.Patient
                .FirstOrDefaultAsync(p => p.PersonalNumber == patient.PersonalNumber && p.Id != patient.Id);

            if (existingPatient != null)
            {
                ModelState.AddModelError("PersonalNumber", "Personal number must be unique.");
            }

            var existingPhoneNumber = await _context.Patient
                .FirstOrDefaultAsync(p => p.PhoneNumber == patient.PhoneNumber && p.Id != patient.Id);

            if (existingPhoneNumber != null)
            {
                ModelState.AddModelError("PhoneNumber", "Phone number must be unique.");
            }

            if (ModelState.IsValid)
            {
                try
                {
                    _context.Update(patient);

                    var user = await _context.Users.FirstOrDefaultAsync(u => u.PatientId == patient.Id);
                    if (user != null)
                    {
                        user.FirstName = patient.FirstName;
                        user.LastName = patient.LastName;
                        _context.Update(user);
                    }

                    await _context.SaveChangesAsync();
                }
                catch (DbUpdateConcurrencyException)
                {
                    if (!PatientExists(patient.Id))
                    {
                        return NotFound();
                    }
                    else
                    {
                        throw;
                    }
                }
                return RedirectToAction(nameof(Index));
            }

            ViewData["DoctorId"] = new SelectList(_context.Doctor, "Id", "FullNameWithId", patient.DoctorId);
            return View(patient);
        }


        // GET: Patients/Delete/5
        public async Task<IActionResult> Delete(int? id)
        {
            if (id == null)
            {
                return NotFound();
            }

            var patient = await _context.Patient
                .Include(p => p.Doctor)
                .FirstOrDefaultAsync(m => m.Id == id);
            if (patient == null)
            {
                return NotFound();
            }

            var doctor = await _context.Doctor.FindAsync(patient.DoctorId);
            ViewBag.DoctorDetails = $"Id: {doctor.Id} - {doctor.FirstName} {doctor.LastName}";

            return View(patient);
        }

        // POST: Patients/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteConfirmed(int id)
        {
            var patient = await _context.Patient.FindAsync(id);
            if (patient != null)
            {
                // CZY ISTNIEJE USER O TYM ID
                var user = await _context.Users.FirstOrDefaultAsync(u => u.PatientId == patient.Id);
                if (user != null)
                {
                    TempData["ErrorMessage"] = "This patient cannot be deleted. Please ensure that their associated account is also deleted first!";
                    return RedirectToAction("Delete");
                }
                _context.Patient.Remove(patient);
                _context.HealthCard.RemoveRange(_context.HealthCard.Where(h => h.PatientId == patient.Id));
                _context.Recording.RemoveRange(_context.Recording.Where(r => r.PatientId == patient.Id));
            }

            await _context.SaveChangesAsync();

            var doctor = await _context.Doctor.FindAsync(patient.DoctorId);
            ViewBag.DoctorDetails = $"Id: {doctor.Id} - {doctor.FirstName} {doctor.LastName}";

            return RedirectToAction(nameof(Index));
        }

        private bool PatientExists(int id)
        {
            return _context.Patient.Any(e => e.Id == id);
        }
    }
}
